"""
Fine-tunes gpt2-medium with a latent autoregression penalty.

The model is divided into three zones:
  Encoder:    transformer.h[0 .. layer_a-1]     (tokens -> latent)
  Latent AR:  transformer.h[layer_a .. layer_b-1]  (latent -> latent)
  Decoder:    transformer.h[layer_b .. end]     (latent -> logits)

The penalty term minimises the L2 distance between:
  h_b[:, i, :]    -- output of the latent AR core at position i
  h_a[:, i+1, :] -- output of the encoder at position i+1
This trains the latent AR core to predict the next latent token, enabling
inference-time autoregression that bypasses the encoder entirely.
"""

import os
import time
from types import SimpleNamespace
from torch.utils.checkpoint import checkpoint as grad_ckpt

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed
from projects.latent_ar.wiki_data import load_train_tokens

# -----------------------------------------------------------------------------
# Config

_DIR = os.path.dirname(os.path.abspath(__file__))

model_type     = 'gpt2-medium'  # 24 layers, 345M params
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_a        = 6      # encoder outputs after this block (1-indexed -> hooks h[5])
layer_b        = 18     # latent AR outputs after this block (1-indexed -> hooks h[17])
lambda_penalty = 1e-3   # penalty weight; tune to keep lambda*penalty ~ ce_loss
block_size     = 1024   # shorter sequences → 4× less attention memory (scales as T²)
batch_size     = 12 if device == 'cuda' else 2
learning_rate  = 3e-5   # conservative LR for fine-tuning
weight_decay   = 0.1
betas          = (0.9, 0.95)
max_iters      = 10000
epochs         = 8
grad_norm_clip = 1.0
log_interval   = 10
ckpt_interval  = 100
ckpt_path = os.path.join(_DIR, 'latent_ar_checkpoint.pt')


class TokenDataset(Dataset):
    """
    Randomly samples block_size-length chunks from a memory-mapped token array.
    __len__ is set to a fixed epoch size so DataLoader never builds a full index
    permutation (which would be ~38 GB for the full Wikipedia corpus).

    Uses a per-instance RNG seeded from system entropy so samples differ
    between runs regardless of any global seed set elsewhere.
    """
    def __init__(self, tokens, block_size, epoch_size):
        self.tokens = tokens
        self.block_size = block_size
        self.max_start = len(tokens) - block_size - 1
        self.epoch_size = epoch_size
        self.rng = np.random.default_rng()  # system-entropy seed

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        pos = int(self.rng.integers(0, self.max_start))
        chunk = torch.from_numpy(
            self.tokens[pos : pos + self.block_size + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def make_hook(buf):
    """Returns a forward hook that stores the block output tensor in buf['act']."""
    def hook(module, input, output):
        buf['act'] = output
    return hook


def train():
    if device == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        torch.set_num_threads(10)
    set_seed(3407)

    # --- data ---
    tokens = load_train_tokens()
    epoch_size = max_iters * batch_size  # one "epoch" = exactly one full training run
    dataset = TokenDataset(tokens, block_size, epoch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Dataset: {len(tokens):,} tokens, epoch_size={epoch_size:,}, block_size={block_size}")

    # --- model ---
    model = GPT.from_pretrained(model_type)
    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        print("No checkpoint found — starting from pretrained weights.")
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = len(model.transformer.h)
    print(f"Model: {model_type}, {n_params/1e6:.1f}M params, {n_layers} transformer blocks")
    assert layer_b <= n_layers, f"layer_b={layer_b} exceeds model depth {n_layers}"

    # --- gradient checkpointing ---
    # Instead of storing all intermediate tensors in all 24 blocks for backprop,
    # only store each block's input and recompute internals during backward.
    # Reduces activation memory from ~4 GB to ~100 MB at the cost of a second
    # forward pass per block during backward (~2× compute, but CPU is the bottleneck).
    for block in model.transformer.h:
        orig_fwd = block.forward
        block.forward = lambda x, f=orig_fwd: grad_ckpt(f, x, use_reentrant=False)

    # --- hooks ---
    # Hooks capture activations after the chosen blocks on every forward pass.
    # h[layer_a - 1] is the last encoder block; h[layer_b - 1] is the last
    # latent-AR block.  Both output tensors have shape (B, T, n_embd).
    h_a_buf, h_b_buf = {}, {}
    handle_a = model.transformer.h[layer_a - 1].register_forward_hook(make_hook(h_a_buf))
    handle_b = model.transformer.h[layer_b - 1].register_forward_hook(make_hook(h_b_buf))
    print(f"Hooks on block indices {layer_a-1} (encoder out) and {layer_b-1} (latent AR out)")

    # --- optimizer ---
    train_cfg = SimpleNamespace(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )
    optimizer = model.configure_optimizers(train_cfg)

    # --- training loop ---
    data_iter = iter(loader)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch}')
        ep_ckpt_path = os.path.join(_DIR, f'latent_ar_checkpoint-{epoch}.pt')
        for iter_num in range(1, max_iters + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # Forward pass.  The hooks populate h_a_buf and h_b_buf as side effects.
            logits, ce_loss = model(x, y)

            # Latent AR penalty.
            # h_b[:, i, :] is the latent AR prediction at position i.
            # h_a[:, i+1, :] is what the encoder actually produces at position i+1.
            # Minimising their L2 distance trains the middle layers to be an
            # autoregressor in latent space.
            h_a = h_a_buf['act']  # (B, T, n_embd)
            h_b = h_b_buf['act']  # (B, T, n_embd)
            penalty = ((h_a[:, 1:, :] - h_b[:, :-1, :]) ** 2).sum(dim=-1).mean()

            loss = ce_loss + lambda_penalty * penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            optimizer.step()

            if iter_num % log_interval == 0:
                t1 = time.time()
                ms_per_iter = (t1 - t0) / log_interval * 1000
                t0 = t1
                penalty_contrib = lambda_penalty * penalty.item()
                ratio = penalty_contrib / (ce_loss.item() + 1e-8)
                ratio_str = f" [penalty/ce ratio: {ratio:.2f}]"
                if ratio > 10:
                    ratio_str += " <- consider reducing lambda_penalty"
                elif ratio < 0.01:
                    ratio_str += " <- consider increasing lambda_penalty"
                print(
                    f"iter {iter_num:6d} | "
                    f"loss {loss.item():.4f} | "
                    f"ce {ce_loss.item():.4f} | "
                    f"penalty {penalty.item():.4f} | "
                    f"{ms_per_iter:.0f}ms/iter"
                    + ratio_str
                )

            if iter_num % ckpt_interval == 0:
                torch.save(model.state_dict(), ep_ckpt_path)
                print(f"Checkpoint saved -> {ep_ckpt_path}")

    torch.save(model.state_dict(), ep_ckpt_path)
    #print(f"Training complete. Checkpoint: {ckpt_path}")

    handle_a.remove()
    handle_b.remove()


def generate_sample():
    """
    Load the fine-tuned checkpoint and run standard token-by-token generation
    to verify that language modelling ability is preserved.
    """
    print("\n--- Generation sample (standard token AR) ---")
    model = GPT.from_pretrained(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    tokenizer = BPETokenizer()
    prompt = "Wikipedia is a free online encyclopedia"
    x = tokenizer(prompt).to(device)  # (1, T)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100, do_sample=True, top_k=40)

    print(tokenizer.decode(y[0]))


if __name__ == '__main__':
    print(f'Device: {device}')
    train()
    generate_sample()
