"""
Measures the mean L2 distance between layer_a and layer_b activations
on a random sample of the training corpus, and compares to mean vector magnitude.

Run from the repo root:
    python -m projects.latent_ar.measure_penalty
"""

import os
import torch
import numpy as np

from mingpt.model import GPT
from projects.latent_ar.wiki_data import load_train_tokens

_DIR = os.path.dirname(os.path.abspath(__file__))

model_type  = 'gpt2-medium'
layer_a     = 6
layer_b     = 18
n_batches   = 20
batch_size  = 2
seq_len     = 1024
ckpt_path   = os.path.join(_DIR, 'latent_ar_checkpoint.pt')


def measure(label, model, tokens):
    rng = np.random.default_rng()

    bufs = [{} for _ in range(24)]
    handles = []
    for i in [layer_a - 1, layer_b - 1]:
        def make_hook(buf):
            def hook(m, inp, out): buf['act'] = out
            return hook
        handles.append(model.transformer.h[i].register_forward_hook(make_hook(bufs[i])))

    norms_a, norms_b, distances = [], [], []

    with torch.no_grad():
        for _ in range(n_batches):
            starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
            x = torch.from_numpy(
                np.stack([tokens[s:s+seq_len].astype(np.int64) for s in starts])
            )
            model(x)
            h_a = bufs[layer_a - 1]['act'].float()
            h_b = bufs[layer_b - 1]['act'].float()

            norms_a.append(h_a.norm(dim=-1).mean().item())
            norms_b.append(h_b.norm(dim=-1).mean().item())
            dist = ((h_a[:, 1:, :] - h_b[:, :-1, :]) ** 2).sum(dim=-1).sqrt().mean().item()
            distances.append(dist)

    for h in handles:
        h.remove()

    mean_norm_a = np.mean(norms_a)
    mean_norm_b = np.mean(norms_b)
    mean_norm   = (mean_norm_a + mean_norm_b) / 2
    mean_dist   = np.mean(distances)

    print(f"\n--- {label} ---")
    print(f"Mean vector magnitude — layer {layer_a} (encoder out): {mean_norm_a:.2f}")
    print(f"Mean vector magnitude — layer {layer_b} (LAR out):     {mean_norm_b:.2f}")
    print(f"Mean L2 distance between paired vectors:               {mean_dist:.2f}")
    print(f"Distance as fraction of mean magnitude:                {mean_dist / mean_norm:.2%}")


if __name__ == '__main__':
    tokens = load_train_tokens()

    print(f"Measuring on {n_batches} batches (batch_size={batch_size}, seq_len={seq_len})")
    print(f"Layer pair: a={layer_a}, b={layer_b}")

    base = GPT.from_pretrained(model_type)
    base.eval()
    measure('Base pretrained model', base, tokens)
    del base

    if os.path.exists(ckpt_path):
        lar = GPT.from_pretrained(model_type)
        lar.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        lar.eval()
        measure('LAR checkpoint', lar, tokens)
    else:
        print(f"\nNo LAR checkpoint found at {ckpt_path}, skipping.")
