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
ckpt_path   = os.path.join(_DIR, 'latent_ar_checkpoint.pt')

LAR_RESULTS_DIR = os.path.join(_DIR, 'lar_results')

# Shared rollout config (applies to all metrics)
PROMPT_LEN   = 128
GEN_STEPS    = 896   # prompt + gen_steps = 1024 (full context)
N_SEQUENCES  = 20


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


def load_lar_model():
    model = GPT.from_pretrained(model_type)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Rollout fidelity

def rollout_fidelity(model, tokens):
    """
    Compare LAR rollout vs. standard AR output distributions over gen_steps
    steps from a prompt of length prompt_len.

    Returns:
        kl_per_step    (GEN_STEPS,) — mean KL(AR||LAR) at each rollout step
        agree_per_step (GEN_STEPS,) — mean top-1 token agreement rate
    """
    rng       = np.random.default_rng(seed=42)
    max_start = len(tokens) - PROMPT_LEN - GEN_STEPS - 1
    kl_sum    = np.zeros(GEN_STEPS)
    agree_sum = np.zeros(GEN_STEPS)

    with torch.no_grad():
        for seq_idx in range(N_SEQUENCES):
            start = int(rng.integers(0, max_start))
            seq = torch.from_numpy(
                tokens[start : start + PROMPT_LEN + GEN_STEPS].astype(np.int64)
            ).unsqueeze(0).to(device)   # (1, prompt_len + gen_steps)

            # --- Standard AR (teacher-forced) ---
            # logits[t] predicts token t+1; relevant slice is
            # logits[prompt_len .. prompt_len+gen_steps-1]
            ar_logits, _ = model(seq)   # (1, prompt_len+gen_steps, vocab)
            ar_probs = F.softmax(
                ar_logits[0, PROMPT_LEN : PROMPT_LEN + GEN_STEPS, :],
                dim=-1,
            )   # (gen_steps, vocab)

            # --- LAR rollout ---
            latents = run_encoder(model, seq[:, :PROMPT_LEN])  # (1, prompt_len, n_embd)

            for step in range(GEN_STEPS):
                lar_out    = run_lar_core(model, latents)               # (1, T, n_embd)
                new_latent = lar_out[:, -1:, :]                         # (1, 1, n_embd)
                latents    = torch.cat([latents, new_latent], dim=1)    # (1, T+1, n_embd)

            # Decode once: causal attention means logits[t] is identical to what
            # we'd get decoding latents[:t+1] alone, so all steps are covered.
            logits = run_decoder(model, latents)   # (1, prompt_len + gen_steps, vocab)
            lar_probs = F.softmax(
                logits[0, PROMPT_LEN : PROMPT_LEN + GEN_STEPS, :],
                dim=-1,
            )   # (gen_steps, vocab)

            for step in range(GEN_STEPS):
                ar_probs_step  = ar_probs[step]
                lar_probs_step = lar_probs[step]

                kl = (ar_probs_step * (ar_probs_step.clamp(min=1e-10).log()
                                       - lar_probs_step.clamp(min=1e-10).log())).sum()
                kl_sum[step]    += kl.item()
                agree_sum[step] += (ar_probs_step.argmax() == lar_probs_step.argmax()).float().item()

            if (seq_idx + 1) % 5 == 0:
                print(f"  [{seq_idx+1}/{N_SEQUENCES}]  "
                      f"mean KL so far: {kl_sum.mean() / (seq_idx+1):.4f}  "
                      f"mean agreement: {agree_sum.mean() / (seq_idx+1):.2%}")

    return kl_sum / N_SEQUENCES, agree_sum / N_SEQUENCES


# ---------------------------------------------------------------------------
# Latent rollout CE

def latent_rollout_ce(model, tokens):
    """
    Encode a prompt, run the LAR core autoregressively for gen_steps steps,
    decode the full latent sequence once, and measure CE at each rollout position.

    Also computes a standard AR baseline (teacher-forced full model) on the
    same sequences for direct comparison.

    Returns:
        ce_per_step      (GEN_STEPS,) — mean CE at each rollout position (LAR)
        base_ce_per_step (GEN_STEPS,) — mean CE at each rollout position (standard AR)
    """
    rng         = np.random.default_rng(seed=42)
    max_start   = len(tokens) - PROMPT_LEN - GEN_STEPS - 1
    ce_sum      = np.zeros(GEN_STEPS)
    base_ce_sum = np.zeros(GEN_STEPS)

    with torch.no_grad():
        for seq_idx in range(N_SEQUENCES):
            start = int(rng.integers(0, max_start))
            # Load prompt + gen_steps tokens as input, +1 for the final target
            seq = torch.from_numpy(
                tokens[start : start + PROMPT_LEN + GEN_STEPS + 1].astype(np.int64)
            ).unsqueeze(0).to(device)   # (1, K + gen_steps + 1)

            # targets: tokens K+1 .. K+gen_steps (ground truth for each rollout position)
            targets = seq[0, PROMPT_LEN + 1 : PROMPT_LEN + 1 + GEN_STEPS]

            # --- Standard AR baseline (teacher-forced) ---
            # Run full model on K+gen_steps tokens; evaluate at positions K..K+gen_steps-1
            base_logits, _ = model(seq[:, : PROMPT_LEN + GEN_STEPS])
            base_ces = F.cross_entropy(
                base_logits[0, PROMPT_LEN : PROMPT_LEN + GEN_STEPS, :],
                targets,
                reduction='none',
            ).cpu().numpy()   # (gen_steps,)
            base_ce_sum += base_ces

            # --- LAR rollout ---
            latents = run_encoder(model, seq[:, :PROMPT_LEN])  # (1, K, n_embd)

            for step in range(GEN_STEPS):
                lar_out    = run_lar_core(model, latents)
                new_latent = lar_out[:, -1:, :]
                latents    = torch.cat([latents, new_latent], dim=1)

            # latents: (1, K + gen_steps, n_embd)
            # Decode once; evaluate CE at positions K..K+gen_steps-1
            logits = run_decoder(model, latents)
            ces = F.cross_entropy(
                logits[0, PROMPT_LEN : PROMPT_LEN + GEN_STEPS, :],
                targets,
                reduction='none',
            ).cpu().numpy()   # (gen_steps,)
            ce_sum += ces

            if (seq_idx + 1) % 5 == 0:
                mean_ce      = ce_sum.mean()      / (seq_idx + 1)
                mean_base_ce = base_ce_sum.mean() / (seq_idx + 1)
                print(f"  [{seq_idx+1}/{N_SEQUENCES}]  "
                      f"mean CE: {mean_ce:.4f}  baseline: {mean_base_ce:.4f}  "
                      f"delta: {mean_ce - mean_base_ce:+.4f}")

    return ce_sum / N_SEQUENCES, base_ce_sum / N_SEQUENCES


# ---------------------------------------------------------------------------
# Latent trajectory divergence

def latent_trajectory_divergence(model, tokens):
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
    rng          = np.random.default_rng(seed=42)
    max_start    = len(tokens) - PROMPT_LEN - GEN_STEPS - 1
    l2_sum       = np.zeros(GEN_STEPS)
    mag_sum      = 0.0

    with torch.no_grad():
        for seq_idx in range(N_SEQUENCES):
            start = int(rng.integers(0, max_start))
            seq = torch.from_numpy(
                tokens[start : start + PROMPT_LEN + GEN_STEPS].astype(np.int64)
            ).unsqueeze(0).to(device)   # (1, K + gen_steps)

            # Ground-truth latent trajectory from the encoder
            ground_truth = run_encoder(model, seq)   # (1, K + gen_steps, n_embd)
            mag_sum += ground_truth.float().norm(dim=-1).mean().item()

            # Seed rollout with real encoder outputs for the prompt
            latents = ground_truth[:, :PROMPT_LEN, :]

            for step in range(GEN_STEPS):
                lar_out   = run_lar_core(model, latents)
                predicted = lar_out[:, -1:, :]                              # (1, 1, n_embd)
                gt        = ground_truth[:, PROMPT_LEN + step : PROMPT_LEN + step + 1, :]

                l2 = (predicted.float() - gt.float()).pow(2).sum(dim=-1).sqrt().item()
                l2_sum[step] += l2

                latents = torch.cat([latents, predicted], dim=1)

            if (seq_idx + 1) % 5 == 0:
                mean_l2 = l2_sum.mean() / (seq_idx + 1)
                print(f"  [{seq_idx+1}/{N_SEQUENCES}]  mean L2 so far: {mean_l2:.4f}")

    l2_per_step    = l2_sum / N_SEQUENCES
    mean_magnitude = mag_sum / N_SEQUENCES
    return l2_per_step, mean_magnitude


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
    if device == 'cpu':
        torch.set_num_threads(10)

    tokens = load_test_tokens()
    model  = load_lar_model()

    # --- Rollout fidelity ---
    print(f"Rollout fidelity")
    print(f"prompt_len={PROMPT_LEN}, gen_steps={GEN_STEPS}, "
          f"n_sequences={N_SEQUENCES}\n")
    t0 = time.time()
    kl_per_step, agree_per_step = rollout_fidelity(model, tokens)
    elapsed = time.time() - t0

    mean_kl    = float(kl_per_step.mean())
    mean_agree = float(agree_per_step.mean())
    print(f"\nMean KL(AR||LAR): {mean_kl:.4f}  |  Mean top-1 agreement: {mean_agree:.2%}  "
          f"({elapsed:.1f}s)")

    save_results('rollout_fidelity', {
        'metric':     'rollout_fidelity',
        'timestamp':  time.strftime('%Y%m%d_%H%M%S'),
        'ckpt_path':  ckpt_path,
        'config': {
            'prompt_len':  PROMPT_LEN,
            'gen_steps':   GEN_STEPS,
            'n_sequences': N_SEQUENCES,
            'layer_a':     layer_a,
            'layer_b':     layer_b,
        },
        'summary': {
            'mean_kl':    mean_kl,
            'mean_agree': mean_agree,
        },
        'per_step': {
            'kl':    kl_per_step.tolist(),
            'agree': agree_per_step.tolist(),
        },
    })

    # --- Latent trajectory divergence ---
    print(f"\nLatent trajectory divergence")
    print(f"prompt_len={PROMPT_LEN}, gen_steps={GEN_STEPS}, "
          f"n_sequences={N_SEQUENCES}\n")
    t0 = time.time()
    l2_per_step, mean_magnitude = latent_trajectory_divergence(model, tokens)
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
            'prompt_len':  PROMPT_LEN,
            'gen_steps':   GEN_STEPS,
            'n_sequences': N_SEQUENCES,
            'layer_a':     layer_a,
            'layer_b':     layer_b,
        },
        'summary': {
            'mean_l2':        mean_l2,
            'l2_step0':       l2_step0,
            'l2_final':       l2_final,
            'mean_magnitude': mean_magnitude,
            'frac_step0':     frac_step0,
            'frac_final':     frac_final,
        },
        'per_step': {
            'l2':      l2_per_step.tolist(),
            'l2_frac': (l2_per_step / mean_magnitude).tolist(),
        },
    })

    # --- Latent rollout CE ---
    print(f"\nLatent rollout CE")
    print(f"prompt_len={PROMPT_LEN}, gen_steps={GEN_STEPS}, "
          f"n_sequences={N_SEQUENCES}\n")
    t0 = time.time()
    ce_per_step, base_ce_per_step = latent_rollout_ce(model, tokens)
    elapsed = time.time() - t0

    mean_ce      = float(ce_per_step.mean())
    mean_base_ce = float(base_ce_per_step.mean())
    print(f"\nMean CE (LAR rollout): {mean_ce:.4f}  |  Mean CE (baseline): {mean_base_ce:.4f}  "
          f"|  Delta: {mean_ce - mean_base_ce:+.4f}  ({elapsed:.1f}s)")

    save_results('latent_rollout_ce', {
        'metric':    'latent_rollout_ce',
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'ckpt_path': ckpt_path,
        'config': {
            'prompt_len':  PROMPT_LEN,
            'gen_steps':   GEN_STEPS,
            'n_sequences': N_SEQUENCES,
            'layer_a':     layer_a,
            'layer_b':     layer_b,
        },
        'summary': {
            'mean_ce':      mean_ce,
            'mean_base_ce': mean_base_ce,
            'mean_delta':   mean_ce - mean_base_ce,
        },
        'per_step': {
            'ce':      ce_per_step.tolist(),
            'base_ce': base_ce_per_step.tolist(),
        },
    })


if __name__ == '__main__':
    run()
