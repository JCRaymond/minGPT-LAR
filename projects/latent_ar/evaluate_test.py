"""
Evaluates the LAR checkpoint against the held-out test set.

Metrics reported:
  CE / perplexity     — standard next-token prediction loss (language modelling quality)
  bypass ablation CE  — encoder output fed directly to decoder (reconstruction)
  shift ablation CE   — oracle-shifted encoder output fed to decoder (reconstruction,
                        last token masked); measures how much work the LAR core does
  L2 / magnitudes     — mean L2 distance between h_a and h_b, and vector magnitudes
  mean bias ‖μ‖       — norm of the mean residual vector; near 0 means errors are unbiased
                        (expressed as a fraction of mean L2 for scale-invariant interpretation)
  baseline delta      — CE difference vs. base pretrained model on the same batches

Run from the repo root:
    python -m projects.latent_ar.evaluate_test
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from mingpt.model import GPT
from projects.latent_ar.wiki_data import load_test_tokens

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config

model_type  = 'gpt2-medium'
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_a     = 6
layer_b     = 18
block_size  = 1024
batch_size  = 12
n_batches   = 100
ckpt_path   = os.path.join(_DIR, 'latent_ar_checkpoint.pt')


# ---------------------------------------------------------------------------

class ShiftLayer(nn.Module):
    """Oracle LAR: output[t] = encoder[t+1], zeros at last position."""
    def forward(self, x):
        out = torch.zeros_like(x)
        out[:, :-1, :] = x[:, 1:, :]
        return out


def make_hook(buf):
    def hook(m, inp, out): buf['act'] = out
    return hook


def sample_batches(tokens, rng):
    max_start = len(tokens) - block_size - 1
    starts = [rng.integers(0, max_start, size=batch_size) for _ in range(n_batches)]
    return starts


def eval_next_token_ce(model, tokens, batch_starts):
    """Standard next-token prediction CE (language modelling quality)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for starts in batch_starts:
            x = torch.from_numpy(
                np.stack([tokens[s : s + block_size].astype(np.int64) for s in starts])
            ).to(device)
            y = torch.from_numpy(
                np.stack([tokens[s + 1 : s + block_size + 1].astype(np.int64) for s in starts])
            ).to(device)
            _, ce = model(x, y)
            total += ce.item()
    return total / n_batches


def eval_reconstruction_ce(model, tokens, batch_starts, ignore_last=False):
    """Reconstruction CE: target = input, optionally masking the last position."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for starts in batch_starts:
            x = torch.from_numpy(
                np.stack([tokens[s : s + block_size].astype(np.int64) for s in starts])
            ).to(device)
            if ignore_last:
                # shift targets left by 1: reconstruct the token that was actually encoded
                y = torch.roll(x, -1, dims=1)
                y[:, -1] = -1  # -1 is the sentinel for ignore_index in F.cross_entropy
            else:
                y = x
            _, ce = model(x, y)
            total += ce.item()
    return total / n_batches


def eval_l2(model, tokens, batch_starts):
    """Mean L2 distance between h_a and h_b activations, and vector magnitudes."""
    model.eval()
    h_a_buf, h_b_buf = {}, {}
    handle_a = model.transformer.h[layer_a - 1].register_forward_hook(make_hook(h_a_buf))
    handle_b = model.transformer.h[layer_b - 1].register_forward_hook(make_hook(h_b_buf))

    norms_a, norms_b, distances = [], [], []
    residual_sum = None
    n_positions  = 0

    with torch.no_grad():
        for starts in batch_starts:
            x = torch.from_numpy(
                np.stack([tokens[s : s + block_size].astype(np.int64) for s in starts])
            ).to(device)
            model(x)
            h_a = h_a_buf['act'].float()
            h_b = h_b_buf['act'].float()
            norms_a.append(h_a.norm(dim=-1).mean().item())
            norms_b.append(h_b.norm(dim=-1).mean().item())
            residuals = h_a[:, 1:, :] - h_b[:, :-1, :]   # (B, T-1, n_embd)
            dist = residuals.pow(2).sum(dim=-1).sqrt().mean().item()
            distances.append(dist)
            # accumulate sum of residuals for mean bias estimate
            batch_sum = residuals.sum(dim=(0, 1)).cpu()   # (n_embd,)
            residual_sum = batch_sum if residual_sum is None else residual_sum + batch_sum
            n_positions += residuals.shape[0] * residuals.shape[1]

    handle_a.remove()
    handle_b.remove()

    mean_residual     = residual_sum / n_positions        # (n_embd,)
    mean_bias         = mean_residual.norm().item()
    mean_l2           = float(np.mean(distances))
    bias_frac         = mean_bias / mean_l2 if mean_l2 > 0 else float('nan')

    return np.mean(norms_a), np.mean(norms_b), mean_l2, mean_bias, bias_frac


def bypass_lar(model):
    for i in range(layer_a, layer_b):
        model.transformer.h[i] = nn.Identity()


def shift_lar(model):
    model.transformer.h[layer_a] = ShiftLayer()
    for i in range(layer_a + 1, layer_b):
        model.transformer.h[i] = nn.Identity()


def load_lar():
    model = GPT.from_pretrained(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def load_base():
    model = GPT.from_pretrained(model_type)
    model.to(device)
    model.eval()
    return model


EVAL_RESULTS_DIR = os.path.join(_DIR, 'eval_results')


def _latest_json():
    """Return the path of the most recent eval JSON, or None."""
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    files = sorted(
        f for f in os.listdir(EVAL_RESULTS_DIR) if f.endswith('.json')
    )
    return os.path.join(EVAL_RESULTS_DIR, files[-1]) if files else None


def _save(path, results):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def run(mode):
    if device == 'cpu':
        torch.set_num_threads(10)

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    if mode == 'cont':
        json_path = _latest_json()
        if json_path is None:
            print("No existing eval file found — starting fresh.")
            mode = 'new'
        else:
            print(f"Resuming from {json_path}")
            with open(json_path) as f:
                saved = json.load(f)
            results = saved['results']

    if mode == 'new':
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(EVAL_RESULTS_DIR, f'eval_{timestamp}.json')
        results = {
            'timestamp': timestamp,
            'ckpt_path': ckpt_path,
            'config': {
                'n_batches': n_batches,
                'batch_size': batch_size,
                'block_size': block_size,
                'layer_a': layer_a,
                'layer_b': layer_b,
            },
        }
        _save(json_path, {'results': results})
        print(f"New eval file: {json_path}")

    tokens = load_test_tokens()
    rng = np.random.default_rng()
    batch_starts = sample_batches(tokens, rng)

    print(f"Test tokens: {len(tokens):,}")
    print(f"Eval: {n_batches} batches x batch_size={batch_size} x block_size={block_size}")
    print(f"LAR layers: h[{layer_a}:{layer_b}]\n")

    def save_result(key, value):
        results[key] = value
        _save(json_path, {'results': results})

    def skip(key):
        if key in results:
            print(f"  (skipping {key} — already computed: {results[key]:.4f})")
            return True
        return False

    # --- LAR: next-token CE / perplexity ---
    if not skip('lar_ce'):
        print("LAR — next-token CE...")
        t0 = time.time()
        model = load_lar()
        save_result('lar_ce', eval_next_token_ce(model, tokens, batch_starts))
        save_result('lar_ppl', math.exp(results['lar_ce']))
        print(f"  CE: {results['lar_ce']:.4f}  PPL: {results['lar_ppl']:.2f}  ({time.time()-t0:.1f}s)")
    else:
        model = load_lar()

    # --- LAR: L2 / magnitudes / bias ---
    if not skip('l2'):
        print("LAR — L2 distance, magnitudes & bias...")
        t0 = time.time()
        norm_a, norm_b, l2, mean_bias, bias_frac = eval_l2(model, tokens, batch_starts)
        save_result('norm_a', norm_a)
        save_result('norm_b', norm_b)
        save_result('l2', l2)
        save_result('mean_bias', mean_bias)
        save_result('bias_frac', bias_frac)
        print(f"  norm_a: {norm_a:.2f}  norm_b: {norm_b:.2f}  L2: {l2:.2f}"
              f"  bias: {mean_bias:.4f} ({bias_frac:.2%} of L2)  ({time.time()-t0:.1f}s)")
    del model

    # --- LAR: bypass ablation ---
    if not skip('bypass_ce'):
        print("LAR — bypass ablation CE...")
        t0 = time.time()
        model = load_lar()
        bypass_lar(model)
        save_result('bypass_ce', eval_reconstruction_ce(model, tokens, batch_starts))
        print(f"  CE: {results['bypass_ce']:.4f}  ({time.time()-t0:.1f}s)")
        del model

    # --- LAR: shift ablation ---
    if not skip('shift_ce'):
        print("LAR — shift ablation CE...")
        t0 = time.time()
        model = load_lar()
        shift_lar(model)
        save_result('shift_ce', eval_reconstruction_ce(model, tokens, batch_starts, ignore_last=True))
        print(f"  CE: {results['shift_ce']:.4f}  ({time.time()-t0:.1f}s)")
        del model

    # --- Base: next-token CE ---
    if not skip('base_ce'):
        print("Base — next-token CE...")
        t0 = time.time()
        model = load_base()
        save_result('base_ce', eval_next_token_ce(model, tokens, batch_starts))
        save_result('base_ppl', math.exp(results['base_ce']))
        print(f"  CE: {results['base_ce']:.4f}  PPL: {results['base_ppl']:.2f}  ({time.time()-t0:.1f}s)")
        del model

    # --- Summary ---
    print(f"\n{'='*52}")
    print(f"  Test set evaluation — {os.path.basename(ckpt_path)}")
    print(f"{'='*52}")
    print(f"  Next-token CE   (LAR):          {results['lar_ce']:.4f}")
    print(f"  Next-token CE   (base):         {results['base_ce']:.4f}")
    print(f"  CE delta vs base:               {results['lar_ce'] - results['base_ce']:+.4f}")
    print(f"  Perplexity      (LAR):          {results['lar_ppl']:.2f}")
    print(f"  Perplexity      (base):         {results['base_ppl']:.2f}")
    print(f"  Mean L2 (h_a vs h_b):          {results['l2']:.4f}")
    print(f"  Mean bias ‖μ‖:                  {results['mean_bias']:.4f} ({results['bias_frac']:.2%} of L2)")
    print(f"  Mean magnitude  (layer {layer_a}):      {results['norm_a']:.2f}")
    print(f"  Mean magnitude  (layer {layer_b}):     {results['norm_b']:.2f}")
    print(f"  Bypass ablation CE:             {results['bypass_ce']:.4f}")
    print(f"  Shift ablation CE:              {results['shift_ce']:.4f}")
    print(f"  Ablation delta (shift-bypass):  {results['shift_ce'] - results['bypass_ce']:+.4f}")
    print(f"\nResults saved to {json_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['new', 'cont'])
    args = parser.parse_args()
    run(args.mode)
