"""
Tests measuring the LAR potential of the model.

Rollout Fidelity — how faithfully does the LAR rollout reproduce standard AR distributions?
    Given a prompt, run standard AR (teacher-forced) and a LAR rollout in
    parallel for gen_steps steps. At each step both methods predict the same
    next token; compare their output distributions via KL divergence and
    top-1 token agreement.

    Standard AR: full model forward on the prompt + ground-truth continuation
    (teacher forcing), giving a distribution at each position.

    LAR rollout: encode the prompt once with the encoder (h[0:layer_a]), then
    autoregressively apply the LAR core (h[layer_a:layer_b]) one step at a
    time — appending only the last position's output to the growing latent
    sequence — and decode with h[layer_b:] + lm_head at each step.

    Alignment: at rollout step s, the LAR latent sequence has length
    prompt_len + s + 1. The decoder at position prompt_len + s predicts
    token prompt_len + s + 1, which is the same token predicted by standard
    AR logits[prompt_len + s].

Latent Trajectory Divergence — how quickly does the LAR rollout drift from the encoder's ground-truth trajectory?
    Run the encoder on the full sequence to get the ground-truth latent trajectory.
    Seed the rollout with real encoder outputs for the prompt, then autoregressively
    apply the LAR core, at each step measuring the L2 distance between the predicted
    latent and the ground-truth encoder output at that position.

    This is entirely decoder-independent — it isolates the LAR core's quality from
    the decoder. L2 at step 0 is the single-step prediction error from real encoder
    inputs (matching the training objective); subsequent steps show how error
    compounds as the LAR core feeds on its own outputs.

    Divergence is also reported as a fraction of mean encoder vector magnitude for
    scale-invariant interpretation.

Latent Rollout CE — how much does language modelling quality degrade under LAR rollout?
    Encode a prompt of prompt_len tokens once, then autoregressively apply the
    LAR core for gen_steps steps without ever re-running the encoder. Decode
    the full latent sequence in a single pass at the end and measure CE at each
    rolled-out position against ground-truth tokens.

    A standard AR baseline (teacher-forced full model) is computed on the same
    sequences, giving a per-step CE comparison. The gap between LAR rollout CE
    and baseline CE measures how much quality degrades as the rollout runs longer.

Run from the repo root:
    python -m projects.latent_ar.lar_test
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mingpt.model import GPT
from projects.latent_ar.wiki_data import load_test_tokens

_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config

model_type  = 'gpt2-medium'
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
layer_a     = 6
layer_b     = 18
#ckpt_path   = os.path.join(_DIR, 'latent_ar_checkpoint.pt')
ckpt_path   = os.path.join(_DIR, 'alarm_gen_checkpoint.pt')

LAR_RESULTS_DIR = os.path.join(_DIR, 'lar_results')
ema_stats_path  = os.path.join(_DIR, 'alarm_ema_stats.pt')

# Shared rollout config (applies to all metrics)
PROMPT_LEN   = 128
GEN_STEPS    = 896   # prompt + gen_steps = 1024 (full context)
N_SEQUENCES  = 30
DECODE_CHUNK = 128   # positions per chunk when applying lm_head — keeps peak VRAM ~400 MB/chunk


# ---------------------------------------------------------------------------
# Model building blocks

def run_encoder(model, idx):
    """Embedding + blocks h[0:layer_a] → latent sequence (B, T, n_embd)."""
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
    x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
    for block in model.transformer.h[:layer_a]:
        x = block(x)
    return x


def run_lar_core(model, x):
    """Blocks h[layer_a:layer_b] → processed latent sequence (B, T, n_embd)."""
    for block in model.transformer.h[layer_a:layer_b]:
        x = block(x)
    return x


def run_decoder(model, x):
    """Blocks h[layer_b:] + ln_f + lm_head → logits (B, T, vocab_size)."""
    for block in model.transformer.h[layer_b:]:
        x = block(x)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def run_decoder_hidden(model, x):
    """Blocks h[layer_b:] + ln_f, without lm_head → hidden states (B, T, n_embd)."""
    for block in model.transformer.h[layer_b:]:
        x = block(x)
    return model.transformer.ln_f(x)


def run_full_model_hidden(model, idx):
    """All 24 blocks + ln_f, without lm_head → hidden states (B, T, n_embd)."""
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
    x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(pos))
    for block in model.transformer.h:
        x = block(x)
    return model.transformer.ln_f(x)


def load_ema_stats():
    return torch.load(ema_stats_path, map_location=device, weights_only=True)


def apply_ema_correction(latent, ema_stats):
    """Map h_b distribution to h_a distribution using bias-corrected EMA stats.
    latent: (B, 1, n_embd)
    """
    decay  = 0.999
    step   = int(ema_stats['ema_step'])
    c      = 1 - decay ** step
    mean_a = ema_stats['ema_mean_a'] / c
    mean_b = ema_stats['ema_mean_b'] / c
    mag_a  = ema_stats['ema_mag_a']  / c
    mag_b  = ema_stats['ema_mag_b']  / c
    return (latent - mean_b) * (mag_a / mag_b) + mean_a


def load_lar_model():
    model = GPT.from_pretrained(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Rollout fidelity

def rollout_fidelity(model, tokens, ema_stats=None):
    """
    Compare LAR rollout vs. standard AR output distributions over gen_steps
    steps from a prompt of length prompt_len.

    Returns:
        kl_per_step    (GEN_STEPS,) — mean KL(AR||LAR) at each rollout step
        agree_per_step (GEN_STEPS,) — mean top-1 token agreement rate
    """
    rng       = np.random.default_rng(seed=42)
    max_start = len(tokens) - PROMPT_LEN - GEN_STEPS - 1
    starts    = rng.integers(0, max_start, size=N_SEQUENCES)

    with torch.no_grad():
        seq = torch.from_numpy(
            np.stack([tokens[s : s + PROMPT_LEN + GEN_STEPS].astype(np.int64) for s in starts])
        ).to(device)   # (N, PROMPT_LEN + GEN_STEPS)

        # --- Standard AR (teacher-forced): run up to ln_f, hold hidden states only ---
        hidden_ar = run_full_model_hidden(model, seq)   # (N, PROMPT_LEN+GEN_STEPS, n_embd)

        # --- LAR rollout ---
        h_a_prompt = run_encoder(model, seq[:, :PROMPT_LEN])  # (N, PROMPT_LEN, n_embd)
        h_b_prompt = run_lar_core(model, h_a_prompt)           # (N, PROMPT_LEN, n_embd)

        # Pre-seed lar_latents with h_b[K-1] at position K so that step 0 produces
        # h_b[K] ≈ h_a[K+1] — matching what the decoder expects at position K.
        lar_latents = torch.cat([h_a_prompt, h_b_prompt[:, -1:, :]], dim=1)  # (N, PROMPT_LEN+1, n_embd)
        dec_latents = h_b_prompt   # decoder uses h_b for prompt positions; generated positions appended below

        log_every = max(1, GEN_STEPS // 4)
        for step in range(GEN_STEPS):
            lar_out    = run_lar_core(model, lar_latents)
            new_latent = lar_out[:, -1:, :]   # h_b[PROMPT_LEN+step] ≈ h_a[PROMPT_LEN+step+1]
            if ema_stats is not None:
                new_latent = apply_ema_correction(new_latent, ema_stats)
            lar_latents = torch.cat([lar_latents, new_latent], dim=1)
            dec_latents = torch.cat([dec_latents, new_latent], dim=1)
            if (step + 1) % log_every == 0:
                print(f"  [step {step+1}/{GEN_STEPS}]")

        # dec_latents[K+s] = h_b[K+s] ≈ h_a[K+s+1] — correct decoder input at position K+s.
        # Run decoder blocks + ln_f only; apply lm_head in chunks below.
        hidden_lar = run_decoder_hidden(model, dec_latents)   # (N, PROMPT_LEN+GEN_STEPS, n_embd)
        del dec_latents, lar_latents

        # Apply lm_head in DECODE_CHUNK-position chunks, covering prompt + generated.
        # Prompt positions (0..PROMPT_LEN-1) use identical decoder inputs for AR and LAR,
        # so KL should be ≈0 and agreement ≈100% there — a built-in sanity check.
        kl_chunks    = []
        agree_chunks = []
        for c in range(0, PROMPT_LEN + GEN_STEPS, DECODE_CHUNK):
            c_end = min(c + DECODE_CHUNK, PROMPT_LEN + GEN_STEPS)
            ar_logits_c  = model.lm_head(hidden_ar[:,  c:c_end, :])
            lar_logits_c = model.lm_head(hidden_lar[:, c:c_end, :])
            ar_p  = F.softmax(ar_logits_c,  dim=-1)
            lar_p = F.softmax(lar_logits_c, dim=-1)
            kl_chunks.append(
                (ar_p * (ar_p.clamp(min=1e-10).log() - lar_p.clamp(min=1e-10).log()))
                .sum(dim=-1).cpu()
            )
            agree_chunks.append((ar_p.argmax(dim=-1) == lar_p.argmax(dim=-1)).float().cpu())

        kl    = torch.cat(kl_chunks,    dim=1)   # (N, PROMPT_LEN + GEN_STEPS)
        agree = torch.cat(agree_chunks, dim=1)   # (N, PROMPT_LEN + GEN_STEPS)
        kl_per_step    = kl.mean(dim=0).numpy()
        agree_per_step = agree.mean(dim=0).numpy()

    return kl_per_step, agree_per_step


# ---------------------------------------------------------------------------
# Latent rollout CE

def latent_rollout_ce(model, tokens, ema_stats=None):
    """
    Encode a prompt, run the LAR core autoregressively for gen_steps steps,
    decode the full latent sequence once, and measure CE at each rollout position.

    Also computes a standard AR baseline (teacher-forced full model) on the
    same sequences for direct comparison.

    Returns:
        ce_per_step      (GEN_STEPS,) — mean CE at each rollout position (LAR)
        base_ce_per_step (GEN_STEPS,) — mean CE at each rollout position (standard AR)
    """
    rng       = np.random.default_rng(seed=42)
    max_start = len(tokens) - PROMPT_LEN - GEN_STEPS - 1
    starts    = rng.integers(0, max_start, size=N_SEQUENCES)

    with torch.no_grad():
        # Load prompt + gen_steps + 1 tokens per sequence
        seq = torch.from_numpy(
            np.stack([tokens[s : s + PROMPT_LEN + GEN_STEPS + 1].astype(np.int64) for s in starts])
        ).to(device)   # (N, PROMPT_LEN + GEN_STEPS + 1)

        # targets for all PROMPT_LEN+GEN_STEPS positions (token at t+1 for each position t)
        targets = seq[:, 1 : PROMPT_LEN + GEN_STEPS + 1]   # (N, PROMPT_LEN+GEN_STEPS)

        # --- Standard AR baseline (teacher-forced): run up to ln_f only ---
        hidden_base = run_full_model_hidden(model, seq[:, :PROMPT_LEN + GEN_STEPS])  # (N, PROMPT_LEN+GEN_STEPS, n_embd)

        # --- LAR rollout ---
        h_a_prompt = run_encoder(model, seq[:, :PROMPT_LEN])  # (N, PROMPT_LEN, n_embd)
        h_b_prompt = run_lar_core(model, h_a_prompt)           # (N, PROMPT_LEN, n_embd)

        # Pre-seed lar_latents with h_b[K-1] at position K so that step 0 produces
        # h_b[K] ≈ h_a[K+1] — matching what the decoder expects at position K.
        lar_latents = torch.cat([h_a_prompt, h_b_prompt[:, -1:, :]], dim=1)  # (N, PROMPT_LEN+1, n_embd)
        dec_latents = h_b_prompt   # decoder uses h_b for prompt positions; generated positions appended below

        log_every = max(1, GEN_STEPS // 4)
        for step in range(GEN_STEPS):
            lar_out    = run_lar_core(model, lar_latents)
            new_latent = lar_out[:, -1:, :]   # h_b[PROMPT_LEN+step] ≈ h_a[PROMPT_LEN+step+1]
            if ema_stats is not None:
                new_latent = apply_ema_correction(new_latent, ema_stats)
            lar_latents = torch.cat([lar_latents, new_latent], dim=1)
            dec_latents = torch.cat([dec_latents, new_latent], dim=1)
            if (step + 1) % log_every == 0:
                print(f"  [step {step+1}/{GEN_STEPS}]")

        # dec_latents[K+s] = h_b[K+s] ≈ h_a[K+s+1] — correct decoder input at position K+s.
        # Run decoder blocks + ln_f only; apply lm_head in chunks below.
        hidden_lar = run_decoder_hidden(model, dec_latents)   # (N, PROMPT_LEN+GEN_STEPS, n_embd)
        del dec_latents, lar_latents

        # Apply lm_head and compute CE in DECODE_CHUNK-position chunks, covering prompt + generated.
        # Prompt positions (0..PROMPT_LEN-1) should show identical CE for base and LAR — sanity check.
        vocab_size = model.lm_head.weight.shape[0]
        base_ce_chunks = []
        lar_ce_chunks  = []
        for c in range(0, PROMPT_LEN + GEN_STEPS, DECODE_CHUNK):
            c_end  = min(c + DECODE_CHUNK, PROMPT_LEN + GEN_STEPS)
            tgts_c = targets[:, c:c_end].reshape(-1)
            base_ce_chunks.append(
                F.cross_entropy(
                    model.lm_head(hidden_base[:, c:c_end, :]).reshape(-1, vocab_size),
                    tgts_c, reduction='none',
                ).reshape(N_SEQUENCES, -1).cpu()
            )
            lar_ce_chunks.append(
                F.cross_entropy(
                    model.lm_head(hidden_lar[:, c:c_end, :]).reshape(-1, vocab_size),
                    tgts_c, reduction='none',
                ).reshape(N_SEQUENCES, -1).cpu()
            )

        base_ce_per_step = torch.cat(base_ce_chunks, dim=1).mean(dim=0).numpy()
        ce_per_step      = torch.cat(lar_ce_chunks,  dim=1).mean(dim=0).numpy()

    mean_ce      = ce_per_step.mean()
    mean_base_ce = base_ce_per_step.mean()
    print(f"  mean CE: {mean_ce:.4f}  baseline: {mean_base_ce:.4f}  "
          f"delta: {mean_ce - mean_base_ce:+.4f}")

    return ce_per_step, base_ce_per_step


# ---------------------------------------------------------------------------
# Latent trajectory divergence

def latent_trajectory_divergence(model, tokens, ema_stats=None):
    """
    Measure how quickly the LAR rollout drifts from the encoder's ground-truth
    latent trajectory.

    At each step the LAR core predicts the next latent from its own previous
    outputs; we compare each prediction to the encoder's actual output at that
    position and record the L2 distance. Step 0 uses real encoder inputs (i.e.
    it measures the training objective directly); subsequent steps accumulate
    exposure bias.

    Returns:
        l2_per_step   (GEN_STEPS,) — mean L2 divergence at each rollout step
        mean_magnitude             — mean encoder vector magnitude (for normalisation)
    """
    rng       = np.random.default_rng(seed=42)
    max_start = len(tokens) - PROMPT_LEN - GEN_STEPS - 1
    starts    = rng.integers(0, max_start, size=N_SEQUENCES)
    l2_sum        = np.zeros(GEN_STEPS)
    pred_norm_sum = np.zeros(GEN_STEPS)
    gt_norm_sum   = np.zeros(GEN_STEPS)

    with torch.no_grad():
        seq = torch.from_numpy(
            np.stack([tokens[s : s + PROMPT_LEN + GEN_STEPS].astype(np.int64) for s in starts])
        ).to(device)   # (N, PROMPT_LEN + GEN_STEPS)

        # Ground-truth latent trajectory from the encoder
        ground_truth   = run_encoder(model, seq)   # (N, PROMPT_LEN + GEN_STEPS, n_embd)
        mean_magnitude = ground_truth.float().norm(dim=-1).mean().item()

        # Seed rollout with real encoder outputs for the prompt
        latents = ground_truth[:, :PROMPT_LEN, :]   # (N, PROMPT_LEN, n_embd)

        log_every = max(1, GEN_STEPS // 4)
        for step in range(GEN_STEPS):
            lar_out   = run_lar_core(model, latents)
            predicted = lar_out[:, -1:, :]   # (N, 1, n_embd)
            if ema_stats is not None:
                predicted = apply_ema_correction(predicted, ema_stats)
            gt        = ground_truth[:, PROMPT_LEN + step : PROMPT_LEN + step + 1, :]

            pred_f = predicted.float()
            gt_f   = gt.float()

            l2 = (pred_f - gt_f).pow(2).sum(dim=-1).sqrt()   # (N, 1)
            l2_sum[step]        += l2.sum().item()
            pred_norm_sum[step] += pred_f.norm(dim=-1).sum().item()
            gt_norm_sum[step]   += gt_f.norm(dim=-1).sum().item()

            latents = torch.cat([latents, predicted], dim=1)

            if (step + 1) % log_every == 0:
                mean_l2 = l2_sum[:step + 1].mean() / N_SEQUENCES
                print(f"  [step {step+1}/{GEN_STEPS}]  mean L2 so far: {mean_l2:.4f}")

    l2_per_step        = l2_sum        / N_SEQUENCES
    pred_norm_per_step = pred_norm_sum / N_SEQUENCES
    gt_norm_per_step   = gt_norm_sum   / N_SEQUENCES
    return l2_per_step, mean_magnitude, pred_norm_per_step, gt_norm_per_step


# ---------------------------------------------------------------------------
# Results persistence

def save_results(name, results):
    os.makedirs(LAR_RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(LAR_RESULTS_DIR, f'{name}_{timestamp}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return path


# ---------------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-correction', action='store_true')
    args = parser.parse_args()

    ema_stats = load_ema_stats() if args.use_correction else None
    if ema_stats is not None:
        print(f"EMA correction enabled (step={int(ema_stats['ema_step'])})")

    if device == 'cpu':
        torch.set_num_threads(10)

    tokens = load_test_tokens()
    model  = load_lar_model()

    # --- Rollout fidelity ---
    print(f"Rollout fidelity")
    print(f"prompt_len={PROMPT_LEN}, gen_steps={GEN_STEPS}, "
          f"n_sequences={N_SEQUENCES}\n")
    t0 = time.time()
    kl_per_step, agree_per_step = rollout_fidelity(model, tokens, ema_stats)
    elapsed = time.time() - t0

    mean_kl_prompt = float(kl_per_step[:PROMPT_LEN].mean())
    mean_kl_gen    = float(kl_per_step[PROMPT_LEN:].mean())
    mean_agree_prompt = float(agree_per_step[:PROMPT_LEN].mean())
    mean_agree_gen    = float(agree_per_step[PROMPT_LEN:].mean())
    print(f"\nPrompt  — KL: {mean_kl_prompt:.4f}  agree: {mean_agree_prompt:.2%}  "
          f"(expect ≈0 / ≈100% — sanity check)")
    print(f"Generated — KL: {mean_kl_gen:.4f}  agree: {mean_agree_gen:.2%}  "
          f"({elapsed:.1f}s)")

    save_results('rollout_fidelity', {
        'metric':     'rollout_fidelity',
        'timestamp':  time.strftime('%Y%m%d_%H%M%S'),
        'ckpt_path':  ckpt_path,
        'config': {
            'prompt_len':     PROMPT_LEN,
            'gen_steps':      GEN_STEPS,
            'n_sequences':    N_SEQUENCES,
            'layer_a':        layer_a,
            'layer_b':        layer_b,
            'use_correction': ema_stats is not None,
        },
        'summary': {
            'mean_kl_prompt':    mean_kl_prompt,
            'mean_kl_gen':       mean_kl_gen,
            'mean_agree_prompt': mean_agree_prompt,
            'mean_agree_gen':    mean_agree_gen,
        },
        'per_step': {
            'kl':    kl_per_step.tolist(),    # length PROMPT_LEN + GEN_STEPS
            'agree': agree_per_step.tolist(),
        },
    })

    # --- Latent trajectory divergence ---
    print(f"\nLatent trajectory divergence")
    print(f"prompt_len={PROMPT_LEN}, gen_steps={GEN_STEPS}, "
          f"n_sequences={N_SEQUENCES}\n")
    t0 = time.time()
    l2_per_step, mean_magnitude, pred_norm_per_step, gt_norm_per_step = \
        latent_trajectory_divergence(model, tokens, ema_stats)
    elapsed = time.time() - t0

    mean_l2       = float(l2_per_step.mean())
    l2_step0      = float(l2_per_step[0])
    l2_final      = float(l2_per_step[-1])
    frac_step0    = l2_step0  / mean_magnitude
    frac_final    = l2_final  / mean_magnitude
    print(f"\nStep 0 L2: {l2_step0:.4f} ({frac_step0:.2%} of magnitude)  |  "
          f"Final L2: {l2_final:.4f} ({frac_final:.2%} of magnitude)  |  "
          f"Mean magnitude: {mean_magnitude:.2f}  ({elapsed:.1f}s)")

    save_results('latent_trajectory_divergence', {
        'metric':    'latent_trajectory_divergence',
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'ckpt_path': ckpt_path,
        'config': {
            'prompt_len':     PROMPT_LEN,
            'gen_steps':      GEN_STEPS,
            'n_sequences':    N_SEQUENCES,
            'layer_a':        layer_a,
            'layer_b':        layer_b,
            'use_correction': ema_stats is not None,
        },
        'summary': {
            'mean_l2':             mean_l2,
            'l2_step0':            l2_step0,
            'l2_final':            l2_final,
            'mean_magnitude':      mean_magnitude,
            'frac_step0':          frac_step0,
            'frac_final':          frac_final,
            'mean_pred_norm':      float(pred_norm_per_step.mean()),
            'mean_gt_norm':        float(gt_norm_per_step.mean()),
        },
        'per_step': {
            'l2':        l2_per_step.tolist(),
            'l2_frac':   (l2_per_step / mean_magnitude).tolist(),
            'pred_norm': pred_norm_per_step.tolist(),
            'gt_norm':   gt_norm_per_step.tolist(),
        },
    })

    # --- Latent rollout CE ---
    print(f"\nLatent rollout CE")
    print(f"prompt_len={PROMPT_LEN}, gen_steps={GEN_STEPS}, "
          f"n_sequences={N_SEQUENCES}\n")
    t0 = time.time()
    ce_per_step, base_ce_per_step = latent_rollout_ce(model, tokens, ema_stats)
    elapsed = time.time() - t0

    mean_ce_prompt    = float(ce_per_step[:PROMPT_LEN].mean())
    mean_ce_gen       = float(ce_per_step[PROMPT_LEN:].mean())
    mean_base_prompt  = float(base_ce_per_step[:PROMPT_LEN].mean())
    mean_base_gen     = float(base_ce_per_step[PROMPT_LEN:].mean())
    print(f"\nPrompt    — LAR CE: {mean_ce_prompt:.4f}  base: {mean_base_prompt:.4f}  "
          f"delta: {mean_ce_prompt - mean_base_prompt:+.4f}  (expect ≈0 — sanity check)")
    print(f"Generated — LAR CE: {mean_ce_gen:.4f}  base: {mean_base_gen:.4f}  "
          f"delta: {mean_ce_gen - mean_base_gen:+.4f}  ({elapsed:.1f}s)")

    save_results('latent_rollout_ce', {
        'metric':    'latent_rollout_ce',
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'ckpt_path': ckpt_path,
        'config': {
            'prompt_len':     PROMPT_LEN,
            'gen_steps':      GEN_STEPS,
            'n_sequences':    N_SEQUENCES,
            'layer_a':        layer_a,
            'layer_b':        layer_b,
            'use_correction': ema_stats is not None,
        },
        'summary': {
            'mean_ce_prompt':   mean_ce_prompt,
            'mean_ce_gen':      mean_ce_gen,
            'mean_base_prompt': mean_base_prompt,
            'mean_base_gen':    mean_base_gen,
            'mean_delta_prompt': mean_ce_prompt - mean_base_prompt,
            'mean_delta_gen':    mean_ce_gen    - mean_base_gen,
        },
        'per_step': {
            'ce':      ce_per_step.tolist(),       # length PROMPT_LEN + GEN_STEPS
            'base_ce': base_ce_per_step.tolist(),
        },
    })


if __name__ == '__main__':
    run()
