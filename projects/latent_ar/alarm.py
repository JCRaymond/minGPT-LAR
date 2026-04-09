"""
Phase 3 — Joint adversarial training for ALARM (Adversarial Latent AutoRegressive Model).

Extends the LAR fine-tuned model with GAN-style adversarial training. The L2 penalty
achieves pointwise alignment (h_b[t] ≈ h_a[t+1]) but doesn't guarantee distributional
alignment. ALARM adds a discriminator trained alongside the LLM to make encoder outputs
(h_a) and LAR core outputs (h_b) statistically indistinguishable.

Three loss terms:
  CE loss      — standard next-token prediction; keeps the model a functioning LM
  LAR penalty  — L2: h_b[t] ≈ h_a[t+1]; pointwise latent autoregression
  Adv loss     — bidirectional LSGAN: distributional alignment between h_a and h_b

Gradient routing:
  Discriminator update: h_a, h_b detached → zero gradient into LLM
  LLM update:
    gen_adv_loss → h_b → LAR core params h[6:18]
    gen_adv_loss → h_b → h_a (input to LAR core) → encoder params h[0:6]
    gen_adv_loss → h_a directly → encoder params h[0:6]
    decoder h[18:24]: downstream of measurement point → CE only

Prerequisites:
  - alarm_gen_checkpoint.pt in projects/latent_ar/ (copy of LAR checkpoint)
  - alarm_disc_checkpoint.pt in DATA_DIR (output of alarm_discriminator.py)

Run from the repo root:
    python -m projects.latent_ar.alarm
"""

import os
import time
from types import SimpleNamespace
from torch.utils.checkpoint import checkpoint as grad_ckpt

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mingpt.model import GPT
from mingpt.utils import set_seed
from projects.latent_ar.alarm_discriminator import Discriminator

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config

model_type     = 'gpt2-medium'
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_a        = 6
layer_b        = 18
lambda_penalty = 8e-3
lambda_adv_target = 0.15   # start conservative; tune up if adv/ce ratio < 0.01
lambda_adv_ramp = False
block_size     = 1024
batch_size     = 12 if device == 'cuda' else 2
learning_rate  = 3e-5
weight_decay   = 0.075
betas          = (0.9, 0.95)
max_iters      = 1000
epochs         = 8
grad_norm_clip = 1.0
log_interval   = 1
ckpt_interval  = 100
disc_lr        = 1e-4
disc_betas     = (0.2, 0.9)
n_critic       = 2    # discriminator updates per LLM update
n_embd         = 1024
DATA_DIR       = '/root'

gen_ckpt_load_path  = os.path.join(_DIR, 'alarm_gen_checkpoint.pt')
gen_ckpt_save_dir   = DATA_DIR
disc_ckpt_load_path = os.path.join(_DIR, 'alarm_disc_checkpoint.pt')
disc_ckpt_save_dir  = DATA_DIR
llm_opt_ckpt_path   = os.path.join(DATA_DIR, 'alarm_llm_opt.pt')
disc_opt_ckpt_path  = os.path.join(DATA_DIR, 'alarm_disc_opt.pt')


# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """
    Randomly samples block_size-length chunks from a memory-mapped token array.
    __len__ is set to a fixed epoch size so DataLoader never builds a full index
    permutation (which would be ~38 GB for the full Wikipedia corpus).
    """
    def __init__(self, tokens, block_size, epoch_size):
        self.tokens = tokens
        self.block_size = block_size
        self.max_start = len(tokens) - block_size - 1
        self.epoch_size = epoch_size
        self.rng = np.random.default_rng()

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        pos = int(self.rng.integers(0, self.max_start))
        chunk = torch.from_numpy(
            self.tokens[pos : pos + self.block_size + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def make_hook(buf):
    def hook(module, input, output): buf['act'] = output
    return hook


def train():
    if device == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        torch.set_num_threads(10)
    set_seed(3407)

    # --- Data ---
    n_tokens = int(np.load(os.path.join(DATA_DIR, 'wiki_tokens_train_meta.npy'))[0])
    tokens   = np.memmap(os.path.join(DATA_DIR, 'wiki_tokens_train.dat'),
                         dtype=np.int32, mode='r', shape=(n_tokens,))
    epoch_size = max_iters * batch_size
    dataset = TokenDataset(tokens, block_size, epoch_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Dataset: {len(tokens):,} tokens, epoch_size={epoch_size:,}, block_size={block_size}")

    # --- LLM ---
    model = GPT.from_pretrained(model_type)
    if os.path.exists(gen_ckpt_load_path):
        print(f"Resuming LLM from {gen_ckpt_load_path}")
        model.load_state_dict(torch.load(gen_ckpt_load_path, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError(
            f"LLM checkpoint not found: {gen_ckpt_load_path}\n"
            "Copy the LAR checkpoint to alarm_gen_checkpoint.pt before running."
        )
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = len(model.transformer.h)
    print(f"Model: {model_type}, {n_params/1e6:.1f}M params, {n_layers} transformer blocks")
    assert layer_b <= n_layers, f"layer_b={layer_b} exceeds model depth {n_layers}"

    # --- Gradient checkpointing ---
    for block in model.transformer.h:
        orig_fwd = block.forward
        block.forward = lambda x, f=orig_fwd: grad_ckpt(f, x, use_reentrant=False)

    # --- Hooks ---
    h_a_buf, h_b_buf = {}, {}
    handle_a = model.transformer.h[layer_a - 1].register_forward_hook(make_hook(h_a_buf))
    handle_b = model.transformer.h[layer_b - 1].register_forward_hook(make_hook(h_b_buf))
    print(f"Hooks on block indices {layer_a-1} (encoder out) and {layer_b-1} (latent AR out)")

    # --- Discriminator ---
    discriminator = Discriminator(n_embd).to(device)
    if os.path.exists(disc_ckpt_load_path):
        print(f"Loading discriminator from {disc_ckpt_load_path}")
        discriminator.load_state_dict(
            torch.load(disc_ckpt_load_path, map_location=device, weights_only=True)
        )
    else:
        raise FileNotFoundError(
            f"Discriminator checkpoint not found: {disc_ckpt_load_path}\n"
            "Run alarm_discriminator.py warm-start first."
        )
    discriminator.train()
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator: {d_params:,} parameters")

    # --- Optimizers ---
    train_cfg = SimpleNamespace(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )
    llm_optimizer  = model.configure_optimizers(train_cfg)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, betas=disc_betas)

    if os.path.exists(llm_opt_ckpt_path):
        print(f"Resuming LLM optimizer from {llm_opt_ckpt_path}")
        llm_optimizer.load_state_dict(torch.load(llm_opt_ckpt_path, map_location=device, weights_only=True))
    if os.path.exists(disc_opt_ckpt_path):
        print(f"Resuming discriminator optimizer from {disc_opt_ckpt_path}")
        disc_optimizer.load_state_dict(torch.load(disc_opt_ckpt_path, map_location=device, weights_only=True))

    # --- Training loop ---
    data_iter = iter(loader)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch}')
        ep_gen_ckpt  = os.path.join(gen_ckpt_save_dir,  f'alarm_gen_checkpoint_{epoch}.pt')
        ep_disc_ckpt = os.path.join(disc_ckpt_save_dir, f'alarm_disc_checkpoint_{epoch}.pt')

        for iter_num in range(1, max_iters + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # --- Forward pass ---
            # Hooks populate h_a_buf and h_b_buf as side effects.
            logits, ce_loss = model(x, y)

            h_a = h_a_buf['act']                   # (B, T, n_embd)
            h_b = h_b_buf['act']
            h_a_flat = h_a.reshape(-1, n_embd)     # (B*T, n_embd)
            h_b_flat = h_b.reshape(-1, n_embd)

            # --- Discriminator update (n_critic steps on same activations) ---
            for _ in range(n_critic):
                disc_optimizer.zero_grad()
                d_real = discriminator(h_a_flat.detach())   # (B*T, 1)
                d_fake = discriminator(h_b_flat.detach())
                disc_loss = (d_real - 1).pow(2).mean() + d_fake.pow(2).mean()
                disc_loss.backward()
                disc_optimizer.step()

            # --- LLM update (CE + L2 + bidirectional LSGAN generator loss) ---
            llm_optimizer.zero_grad(set_to_none=True)

            # Disable discriminator param gradients during LLM backward — they're
            # not needed and computing them would waste memory and time.
            discriminator.requires_grad_(False)

            # No detach: gradients flow from adversarial loss into encoder (via h_a)
            # and into the LAR core (via h_b).
            d_real_gen = discriminator(h_a_flat)        # (B*T, 1)
            d_fake_gen = discriminator(h_b_flat)

            # Bidirectional LSGAN generator loss:
            #   (d_fake_gen - 1)²: push h_b toward looking real (LAR core → encoder dist.)
            #   d_real_gen²:       push h_a toward looking fake (encoder → LAR core dist.)
            # Equilibrium: D outputs ~0.5 for both; distributions are indistinguishable.
            gen_adv_loss = (d_fake_gen - 1).pow(2).mean()# + d_real_gen.pow(2).mean()

            # LAR penalty — same as latent_ar.py
            penalty = ((h_a[:, 1:, :] - h_b[:, :-1, :]) ** 2).sum(dim=-1).mean()

            lambda_adv = lambda_adv_target
            if lambda_adv_ramp:
                total_iter = (epoch-1)*max_iters + iter_num - 1
                lambda_adv *= 1/(1 + np.exp(-0.02*(total_iter-300)))

            total_loss = ce_loss + lambda_penalty * penalty + lambda_adv * gen_adv_loss
            total_loss.backward()

            discriminator.requires_grad_(True)  # restore for next discriminator update

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            llm_optimizer.step()

            if iter_num % log_interval == 0:
                t1 = time.time()
                ms_per_iter = (t1 - t0) / log_interval * 1000
                t0 = t1

                with torch.no_grad():
                    disc_acc = ((d_real > 0.5).float().mean().item() + (d_fake < 0.5).float().mean().item())/2

                penalty_ratio = (lambda_penalty * penalty.item()) / (ce_loss.item() + 1e-8)
                adv_ratio     = (lambda_adv * gen_adv_loss.item()) / (ce_loss.item() + 1e-8)

                ratio_notes = []
                if penalty_ratio > 10:
                    ratio_notes.append("penalty/ce high — consider reducing lambda_penalty")
                elif penalty_ratio < 0.01:
                    ratio_notes.append("penalty/ce low — consider increasing lambda_penalty")
                if adv_ratio > 2.0:
                    ratio_notes.append("adv/ce high — consider reducing lambda_adv")
                elif adv_ratio < 0.01:
                    ratio_notes.append("adv/ce low — consider increasing lambda_adv")

                suffix = f" [{'; '.join(ratio_notes)}]" if ratio_notes else ""
                print(
                    f"iter {iter_num:6d} | "
                    f"loss {total_loss.item():.4f} | "
                    f"ce {ce_loss.item():.4f} | "
                    f"penalty {penalty.item():.4f} | "
                    f"gen_adv {gen_adv_loss.item():.4f} | "
                    f"disc_loss {disc_loss.item():.4f} | "
                    f"disc_acc {disc_acc:.2%} | "
                    f"{ms_per_iter:.0f}ms/iter"
                    f" [penalty/ce: {penalty_ratio:.2f}] [adv/ce: {adv_ratio:.2f}]"
                    + suffix
                )

            if iter_num % ckpt_interval == 0:
                torch.save(model.state_dict(), ep_gen_ckpt)
                torch.save(discriminator.state_dict(), ep_disc_ckpt)
                torch.save(llm_optimizer.state_dict(),  llm_opt_ckpt_path)
                torch.save(disc_optimizer.state_dict(), disc_opt_ckpt_path)
                print(f"Checkpoints saved -> {ep_gen_ckpt}, {ep_disc_ckpt}")

        torch.save(model.state_dict(), ep_gen_ckpt)
        torch.save(discriminator.state_dict(), ep_disc_ckpt)
        torch.save(llm_optimizer.state_dict(),  llm_opt_ckpt_path)
        torch.save(disc_optimizer.state_dict(), disc_opt_ckpt_path)
        print(f"Epoch {epoch} complete. Checkpoints: {ep_gen_ckpt}, {ep_disc_ckpt}")

    handle_a.remove()
    handle_b.remove()


if __name__ == '__main__':
    print(f'Device: {device}')
    train()
