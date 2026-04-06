"""
Ablation: encoder-decoder autoencoder tests.

Two ablations are run back to back:

  bypass — replaces LAR layers (h[layer_a:layer_b]) with nn.Identity, connecting
           the encoder output directly to the decoder input. CE is measured as
           reconstruction (target = input).

  shift  — replaces LAR layers with an ideal oracle that shifts encoder
           representations forward by 1 position (output[t] = encoder[t+1]),
           which is exactly what a perfect LAR core would produce. The last
           position is masked out of CE since it has no valid oracle value.

If the model has successfully pushed all autoregressive logic into the LAR layers,
both tests should yield similar CE. A large gap means the LAR core is still doing
meaningful work that the encoder/decoder pair alone cannot replicate.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from mingpt.model import GPT
from mingpt.utils import set_seed
from projects.latent_ar.wiki_data import load_train_tokens

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config

model_type  = 'gpt2-medium'
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_a     = 6
layer_b     = 18
block_size  = 256   # shorter than training for speed; scales attention as T²
batch_size  = 2
n_batches   = 50
ckpt_path   = os.path.join(_DIR, 'latent_ar_checkpoint.pt')


# ---------------------------------------------------------------------------

class ShiftLayer(nn.Module):
    """Oracle LAR: output[t] = encoder[t+1], zeros at last position."""
    def forward(self, x):
        out = torch.zeros_like(x)
        out[:, :-1, :] = x[:, 1:, :]
        return out


def eval_ce(model, tokens, rng, ignore_last=False):
    """Evaluate mean reconstruction CE over n_batches random chunks."""
    model.eval()
    max_start = len(tokens) - block_size
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            starts = rng.integers(0, max_start, size=batch_size)
            x = torch.stack([
                torch.from_numpy(tokens[s : s + block_size].astype(np.int64))
                for s in starts
            ]).to(device)
            if ignore_last:
                # shift targets left by 1: reconstruct the token that was actually encoded
                y = torch.roll(x, -1, dims=1)
                y[:, -1] = -1  # -1 is the sentinel for ignore_index in F.cross_entropy
            else:
                y = x
            _, ce = model(x, y)
            total += ce.item()
    return total / n_batches


def bypass_lar(model):
    for i in range(layer_a, layer_b):
        model.transformer.h[i] = nn.Identity()


def shift_lar(model):
    model.transformer.h[layer_a] = ShiftLayer()
    for i in range(layer_a + 1, layer_b):
        model.transformer.h[i] = nn.Identity()


def load_model(mode):
    model = GPT.from_pretrained(model_type)
    if mode == 'lar':
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    return model


def run(mode):
    if device == 'cpu':
        torch.set_num_threads(10)
    set_seed(3407)

    tokens = load_train_tokens()
    print(
        f"Tokens: {len(tokens):,} | block_size={block_size}, "
        f"batch_size={batch_size}, n_batches={n_batches}"
    )
    label = 'LAR checkpoint' if mode == 'lar' else 'base pretrained model'
    print(f"Model: {label}")
    print(f"LAR layers: h[{layer_a}:{layer_b}] ({layer_b - layer_a} blocks)\n")

    rng = np.random.default_rng(seed=42)

    # --- bypass ---
    print("Loading model (bypass)...")
    model = load_model(mode)
    bypass_lar(model)
    t0 = time.time()
    bypass_ce = eval_ce(model, tokens, rng, ignore_last=False)
    print(f"Bypass CE:  {bypass_ce:.4f}  ({time.time()-t0:.1f}s)")
    del model

    rng = np.random.default_rng(seed=42)  # same batches for fair comparison

    # --- shift ---
    print("Loading model (shift)...")
    model = load_model(mode)
    shift_lar(model)
    t0 = time.time()
    shift_ce = eval_ce(model, tokens, rng, ignore_last=True)
    print(f"Shift CE:   {shift_ce:.4f}  ({time.time()-t0:.1f}s)")
    del model

    print(f"\nDelta (shift - bypass): {shift_ce - bypass_ce:+.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=['lar', 'orig'], default='lar')
    args = parser.parse_args()
    run(args.mode)
