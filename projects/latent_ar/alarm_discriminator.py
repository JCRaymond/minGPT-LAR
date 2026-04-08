"""
Phase 2 — Discriminator warm-start for ALARM.

Trains a small MLP discriminator on frozen LAR model activations to distinguish
encoder outputs (h_a, "real") from LAR core outputs (h_b, "fake"). The warm-started
discriminator provides a useful adversarial signal from step 1 of Phase 3 (alarm.py)
rather than being random noise.

Architecture: Linear 1024→512→256→1, LeakyReLU(0.2), no sigmoid.
Loss: LSGAN — (D(real) - 1)² + D(fake)² — raw unbounded scores, never saturates.

Run from the repo root:
    python -m projects.latent_ar.alarm_discriminator
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from mingpt.model import GPT

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config

model_type   = 'gpt2-medium'
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_a      = 6
layer_b      = 18
block_size   = 1024
batch_size   = 12 if device == 'cuda' else 2
n_iters      = 500
log_interval = 10
n_embd       = 1024
DATA_DIR     = '/root'

gen_ckpt_path  = os.path.join(_DIR, 'latent_ar_checkpoint.pt')
disc_ckpt_path = os.path.join(DATA_DIR, 'alarm_disc_checkpoint.pt')


# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    Per-position MLP discriminator for LSGAN on latent vectors.

    Input: (N, n_embd) — individual latent vectors (positions flattened across batch)
    Output: (N, 1) — raw scalar score; real target=1, fake target=0, no sigmoid.

    Design:
    - LeakyReLU(0.2): avoids dying-neuron problem of ReLU in discriminator settings.
    - No BatchNorm: computes statistics across real+fake in same batch — harmful here.
    - No final activation: LSGAN is built for raw unbounded outputs; sigmoid would
      saturate and kill gradients when the discriminator is confident.
    """
    def __init__(self, n_embd=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)  # (N, 1), raw scalar


def make_hook(buf):
    def hook(m, inp, out): buf['act'] = out
    return hook


def sample_batch(tokens, rng):
    max_start = len(tokens) - block_size - 1
    starts = rng.integers(0, max_start, size=batch_size)
    return torch.from_numpy(
        np.stack([tokens[s : s + block_size].astype(np.int64) for s in starts])
    ).to(device)


def train():
    if device == 'cpu':
        torch.set_num_threads(10)

    # --- LLM (frozen) ---
    print(f"Loading LAR model from {gen_ckpt_path} ...")
    model = GPT.from_pretrained(model_type)
    model.load_state_dict(torch.load(gen_ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    h_a_buf, h_b_buf = {}, {}
    handle_a = model.transformer.h[layer_a - 1].register_forward_hook(make_hook(h_a_buf))
    handle_b = model.transformer.h[layer_b - 1].register_forward_hook(make_hook(h_b_buf))

    # --- Discriminator ---
    discriminator = Discriminator(n_embd).to(device)
    n_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator: {n_params:,} parameters")

    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # --- Data ---
    n_tokens = int(np.load(os.path.join(DATA_DIR, 'wiki_tokens_train_meta.npy'))[0])
    tokens   = np.memmap(os.path.join(DATA_DIR, 'wiki_tokens_train.dat'),
                         dtype=np.int32, mode='r', shape=(n_tokens,))
    rng = np.random.default_rng()
    print(f"Train tokens: {len(tokens):,}")
    print(f"Device: {device}  batch_size: {batch_size}  n_iters: {n_iters}\n")

    t0 = time.time()
    for iter_num in range(1, n_iters + 1):
        x = sample_batch(tokens, rng)

        with torch.no_grad():
            model(x)

        h_a = h_a_buf['act'].detach()              # (B, T, n_embd)
        h_b = h_b_buf['act'].detach()

        h_a_flat = h_a.reshape(-1, n_embd)         # (B*T, n_embd)
        h_b_flat = h_b.reshape(-1, n_embd)

        d_real = discriminator(h_a_flat)            # (B*T, 1)
        d_fake = discriminator(h_b_flat)

        # LSGAN discriminator loss: real→1, fake→0
        disc_loss = (d_real - 1).pow(2).mean() + d_fake.pow(2).mean()

        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        if iter_num % log_interval == 0:
            t1 = time.time()
            ms_per_iter = (t1 - t0) / log_interval * 1000
            t0 = t1

            with torch.no_grad():
                acc = ((d_real > 0.5) & (d_fake < 0.5)).float().mean().item()

            print(
                f"iter {iter_num:5d} | "
                f"disc_loss {disc_loss.item():.4f} | "
                f"disc_acc {acc:.2%} | "
                f"{ms_per_iter:.0f}ms/iter"
            )

            torch.save(discriminator.state_dict(), disc_ckpt_path)

    torch.save(discriminator.state_dict(), disc_ckpt_path)
    print(f"\nWarm-start complete. Discriminator saved to {disc_ckpt_path}")

    handle_a.remove()
    handle_b.remove()


if __name__ == '__main__':
    train()
