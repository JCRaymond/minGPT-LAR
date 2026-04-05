"""
Ablation: encoder-decoder autoencoder test.

Bypasses the LAR layers (transformer.h[layer_a : layer_b]) by replacing them
with nn.Identity, connecting the encoder output directly to the decoder input.
Cross-entropy is measured between the input tokens and the output logits
(reconstruction), not next-token prediction. No gradient updates.

A low reconstruction CE indicates the encoder (h[0:layer_a]) and decoder
(h[layer_b:]) have developed complementary representations.
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

def eval_ce(model, tokens, rng):
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
            _, ce = model(x, x)  # target = input (reconstruction, not next-token)
            total += ce.item()
    return total / n_batches


def bypass_lar(model):
    for i in range(layer_a, layer_b):
        model.transformer.h[i] = nn.Identity()


def run(mode):
    if device == 'cpu':
        torch.set_num_threads(10)
    set_seed(3407)

    tokens = load_train_tokens()
    print(
        f"Tokens: {len(tokens):,} | block_size={block_size}, "
        f"batch_size={batch_size}, n_batches={n_batches}"
    )
    print(f"Bypassing h[{layer_a}:{layer_b}] ({layer_b - layer_a} of 24 blocks skipped)\n")

    print(f"Loading {'LAR checkpoint' if mode == 'lar' else 'base pretrained model'}...")
    model = GPT.from_pretrained(model_type)
    if mode == 'lar':
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    bypass_lar(model)
    t0 = time.time()
    ce = eval_ce(model, tokens, np.random.default_rng(seed=42))
    print(f"Ablated reconstruction CE ({mode}): {ce:.4f}  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=['lar', 'orig'], default='lar')
    args = parser.parse_args()
    run(args.mode)
