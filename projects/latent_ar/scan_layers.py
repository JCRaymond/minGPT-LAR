"""
Scans all (a, b) layer pairs in gpt2-medium to measure the latent-AR penalty
for each pair across a large random sample of the Wikipedia corpus.

For every pair (a, b) with a < b, the penalty is the per-element MSE between:
  h_b[:, i, :]   -- output of layer b at position i
  h_a[:, i+1, :] -- output of layer a at position i+1

A lower value means those two layers are already more compatible as
encoder/latent-AR endpoints, requiring less training to bridge the gap.

Results are accumulated as a running mean so the scan can run overnight and be
interrupted cleanly at any time with Ctrl+C.  Progress is checkpointed every
500 batches to layer_scan_checkpoint.npz.  A final ranked table and heatmap
(if matplotlib is available) are printed/saved at the end.

Run from the repo root:
    python -m projects.latent_ar.scan_layers orig   # scan base gpt2-medium weights
    python -m projects.latent_ar.scan_layers lar    # scan LAR fine-tuned checkpoint
"""

import os
import signal
import sys
import time

import numpy as np
import torch

from mingpt.model import GPT
from mingpt.utils import set_seed
from projects.latent_ar.wiki_data import load_train_tokens

# -----------------------------------------------------------------------------
# Config

_DIR = os.path.dirname(os.path.abspath(__file__))

model_type    = 'gpt2-medium'   # 24 layers
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
seq_len       = 1024            # matches training block_size
batch_size    = 4 if device == 'cuda' else 2
save_interval = 500             # checkpoint every N batches
log_interval  = 100

RESULTS_DIR = os.path.join(_DIR, 'layer_scan_results')


def _run_paths(mode):
    """
    Return (latent_ar_ckpt, checkpoint_path, plot_prefix) for the given mode.

    orig  — all runs share a single checkpoint and overwrite the same plots,
           since the base model never changes.
    lar   — each run gets its own timestamped subdirectory so results from
           different training stages can be compared.
    alarm — each run gets its own timestamped subdirectory so results from
           different training stages can be compared.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if mode == 'orig':
        return (
            None,
            os.path.join(RESULTS_DIR, 'orig_checkpoint.npz'),
            os.path.join(RESULTS_DIR, 'orig'),
        )
    elif mode == 'lar':
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(RESULTS_DIR, f'layer_scan_lar_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)
        return (
            os.path.join(_DIR, 'latent_ar_checkpoint.pt'),
            os.path.join(run_dir, 'checkpoint.npz'),
            os.path.join(run_dir, 'lar'),
        )
    elif mode == 'alarm':
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(RESULTS_DIR, f'layer_scan_alarm_{timestamp}')
        os.makedirs(run_dir, exist_ok=True)
        return (
            os.path.join(_DIR, 'alarm_gen_checkpoint.pt'),
            os.path.join(run_dir, 'checkpoint.npz'),
            os.path.join(run_dir, 'lar'),
        )
    else:
        print(f"ERROR: unknown mode '{mode}'. Use 'orig', 'lar', or 'alarm'.")
        sys.exit(1)

# -----------------------------------------------------------------------------

def make_hook(buf):
    def hook(module, input, output):
        buf['act'] = output
    return hook


def scan(mode):
    latent_ar_ckpt, checkpoint_path, plot_prefix = _run_paths(mode)

    if device == 'cpu':
        torch.set_num_threads(10)
    set_seed(3407)
    tokens = load_train_tokens()
    n_tokens = len(tokens)

    print(f"Mode: {mode}")
    print(f"Loading {model_type}...")
    model = GPT.from_pretrained(model_type)
    if latent_ar_ckpt is not None and os.path.exists(latent_ar_ckpt):
        print(f"Loading LAR weights from {latent_ar_ckpt}")
        model.load_state_dict(torch.load(latent_ar_ckpt, map_location=device, weights_only=True))
    else:
        print("Scanning base pretrained model (no LAR checkpoint loaded).")
    model.to(device)
    model.eval()
    n_layers = len(model.transformer.h)
    n_embd   = model.transformer.h[0].mlp.c_proj.out_features
    n_pairs  = n_layers * (n_layers - 1) // 2
    print(f"Model: {n_layers} layers, n_embd={n_embd}, device={device}")
    print(f"Pairs to track: {n_pairs}")

    # Register a hook on every transformer block.
    layer_bufs = [{} for _ in range(n_layers)]
    handles = []
    for i in range(n_layers):
        h = model.transformer.h[i].register_forward_hook(make_hook(layer_bufs[i]))
        handles.append(h)

    # Load existing checkpoint or start fresh.
    if os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        penalty_sum  = ckpt['penalty_sum'].copy()
        cos_sum      = (ckpt['cos_sum'].copy() if 'cos_sum' in ckpt
                        else np.zeros((n_layers, n_layers), dtype=np.float64))
        pair_counts  = (ckpt['pair_counts'].copy() if 'pair_counts' in ckpt
                        else np.full((n_layers, n_layers),
                                     int(ckpt['total_pairs'].item()), dtype=np.float64))
        total_pairs  = int(ckpt['total_pairs'].item())
        batch_offset = int(ckpt['batch_num'].item())
        # Repair corrupted entries: NaN, Inf, or negative (MSE is always >= 0).
        bad_mask = ~np.isfinite(penalty_sum) | (penalty_sum < 0)
        if bad_mask.any():
            print(f"  Repairing {bad_mask.sum()} corrupted entries in penalty_sum.")
            penalty_sum[bad_mask] = 0.0
            pair_counts[bad_mask] = 0.0
        print(f"Resuming from checkpoint: {total_pairs:,} position pairs, "
              f"{batch_offset:,} batches processed.")
    else:
        penalty_sum  = np.zeros((n_layers, n_layers), dtype=np.float64)
        cos_sum      = np.zeros((n_layers, n_layers), dtype=np.float64)
        pair_counts  = np.zeros((n_layers, n_layers), dtype=np.float64)
        total_pairs  = 0
        batch_offset = 0
        print("Starting fresh scan.")

    # Graceful Ctrl+C: finish the current batch, save, report.
    interrupted = False
    def _handle_sigint(sig, frame):
        nonlocal interrupted
        print("\nCtrl+C received — finishing current batch then saving...")
        interrupted = True
    signal.signal(signal.SIGINT, _handle_sigint)

    rng = np.random.default_rng()  # system-entropy seed — different samples every run
    batch_num = batch_offset
    n_pos_per_batch = batch_size * (seq_len - 1)
    t0 = time.time()
    t_last_log = t0

    print(f"\nScanning — press Ctrl+C to stop cleanly.")
    print(f"Logging every {log_interval} batches "
          f"({log_interval * n_pos_per_batch:,} position pairs per update).\n")

    with torch.no_grad():
        while not interrupted:
            # ---- sample a random batch from the token memmap ---------------
            max_start = n_tokens - seq_len - 1
            starts = rng.integers(0, max_start, size=batch_size)
            chunks = np.stack([
                tokens[s : s + seq_len + 1].astype(np.int64) for s in starts
            ])                                    # (B, seq_len+1)
            x = torch.from_numpy(chunks[:, :-1]).to(device)   # (B, seq_len)

            # ---- forward pass — hooks populate layer_bufs ------------------
            model(x)

            # ---- move activations to CPU immediately to free VRAM ----------
            # Keeping all 24 tensors on GPU while doing 276 pair comparisons
            # spikes memory and can crash the ROCm driver on desktop GPUs.
            h_fwd = [layer_bufs[i]['act'][:, 1:, :].cpu().float()
                     for i in range(n_layers)]
            h_bwd = [layer_bufs[i]['act'][:, :-1, :].cpu().float()
                     for i in range(n_layers)]
            for buf in layer_bufs:
                buf.clear()
            del x
            if device == 'cuda':
                torch.cuda.empty_cache()

            # ---- accumulate L2 and cosine for every (a, b) pair (on CPU) ----
            for a in range(n_layers):
                for b in range(a + 1, n_layers):
                    diff = h_fwd[a] - h_bwd[b]
                    mse  = (diff * diff).mean().item()
                    if np.isfinite(mse):
                        penalty_sum[a, b] += mse * n_pos_per_batch
                        pair_counts[a, b] += n_pos_per_batch

                    dot  = (h_fwd[a] * h_bwd[b]).sum(dim=-1)
                    norm = h_fwd[a].norm(dim=-1) * h_bwd[b].norm(dim=-1)
                    cos  = (dot / norm.clamp(min=1e-8)).mean().item()
                    if np.isfinite(cos):
                        cos_sum[a, b] += cos * n_pos_per_batch

            total_pairs += n_pos_per_batch
            batch_num   += 1

            # ---- logging ---------------------------------------------------
            if batch_num % log_interval == 0:
                now       = time.time()
                elapsed   = now - t0
                batches_done = batch_num - batch_offset
                rate      = batches_done / elapsed          # batches/sec
                print(f"  [{elapsed/3600:5.2f}h]  "
                      f"batch {batch_num:,}  |  "
                      f"{total_pairs/1e6:.2f}M position pairs  |  "
                      f"{rate:.2f} batches/sec")

            # ---- periodic checkpoint ---------------------------------------
            if batch_num % save_interval == 0:
                _save_checkpoint(checkpoint_path, penalty_sum, cos_sum, pair_counts, total_pairs, batch_num)

    # Final save + report
    _save_checkpoint(checkpoint_path, penalty_sum, cos_sum, pair_counts, total_pairs, batch_num)
    _report(plot_prefix, penalty_sum, cos_sum, pair_counts, total_pairs, n_layers)

    for h in handles:
        h.remove()


def _save_checkpoint(checkpoint_path, penalty_sum, cos_sum, pair_counts, total_pairs, batch_num):
    np.savez(checkpoint_path,
             penalty_sum=penalty_sum,
             cos_sum=cos_sum,
             pair_counts=pair_counts,
             total_pairs=np.array([total_pairs]),
             batch_num=np.array([batch_num]))
    print(f"  [checkpoint: {total_pairs:,} pairs, {batch_num:,} batches]")


def _report(plot_prefix, penalty_sum, cos_sum, pair_counts, total_pairs, n_layers):
    if total_pairs == 0:
        print("No data collected.")
        return

    safe_counts = np.where(pair_counts > 0, pair_counts, np.nan)
    means    = penalty_sum / safe_counts   # (n_layers, n_layers)
    cos_mean    = cos_sum / total_pairs          # (n_layers, n_layers)
    cos_sq_mean = cos_mean ** 2                  # squared for better resolution near 1

    print(f"\n{'='*62}")
    print(f"Layer pair scan — {total_pairs:,} position pairs accumulated")
    print(f"Metric: per-element MSE  [ h_b[i] vs h_a[i+1] ]")
    print(f"Lower = more compatible as latent-AR (a, b) split points")
    print(f"{'='*62}")

    # Build sorted list of all valid (a < b) pairs
    pairs = [
        (means[a, b], a + 1, b + 1, b - a)   # 1-indexed for display
        for a in range(n_layers)
        for b in range(a + 1, n_layers)
    ]
    pairs.sort()

    n_show = min(30, len(pairs))
    print(f"\n{'Rank':>4}  {'a':>4}  {'b':>4}  {'gap':>4}  {'per-elem MSE':>14}")
    print("─" * 42)
    for rank, (mse, a, b, gap) in enumerate(pairs[:n_show], 1):
        print(f"{rank:>4}  {a:>4}  {b:>4}  {gap:>4}  {mse:>14.6f}")

    print(f"\n  ... {len(pairs)} total pairs")

    # Also print the best pair for each gap size (1, 6, 12, 18, …)
    best_by_gap = {}
    for mse, a, b, gap in pairs:
        if gap not in best_by_gap:
            best_by_gap[gap] = (mse, a, b)

    print(f"\nBest pair per gap size:")
    print(f"{'gap':>4}  {'a':>4}  {'b':>4}  {'per-elem MSE':>14}")
    print("─" * 30)
    for gap in sorted(best_by_gap):
        mse, a, b = best_by_gap[gap]
        print(f"{gap:>4}  {a:>4}  {b:>4}  {mse:>14.6f}")

    # Cosine² similarity table
    cos_pairs = [
        (cos_sq_mean[a, b], a + 1, b + 1, b - a)
        for a in range(n_layers)
        for b in range(a + 1, n_layers)
    ]
    cos_pairs.sort(reverse=True)   # higher cos² = more compatible

    print(f"\n{'='*62}")
    print(f"Cosine² similarity  [ h_b[i] vs h_a[i+1] ]")
    print(f"Higher = more directionally compatible as latent-AR split points")
    print(f"{'='*62}")
    print(f"\n{'Rank':>4}  {'a':>4}  {'b':>4}  {'gap':>4}  {'cos² sim':>10}")
    print("─" * 36)
    for rank, (cos_sq, a, b, gap) in enumerate(cos_pairs[:30], 1):
        print(f"{rank:>4}  {a:>4}  {b:>4}  {gap:>4}  {cos_sq:>10.6f}")

    best_cos_by_gap = {}
    for cos_sq, a, b, gap in cos_pairs:
        if gap not in best_cos_by_gap:
            best_cos_by_gap[gap] = (cos_sq, a, b)

    print(f"\nBest cos² pair per gap size:")
    print(f"{'gap':>4}  {'a':>4}  {'b':>4}  {'cos² sim':>10}")
    print("─" * 26)
    for gap in sorted(best_cos_by_gap):
        cos_sq, a, b = best_cos_by_gap[gap]
        print(f"{gap:>4}  {a:>4}  {b:>4}  {cos_sq:>10.6f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import LogNorm

        max_gap = n_layers - 1
        gap_cmap = matplotlib.colormaps['plasma'].resampled(max_gap)
        ticks = list(range(1, n_layers + 1))

        # --- L2 line plot (log scale) ----------------------------------------
        l2_line_path = f'{plot_prefix}_l2_lines.png'
        fig, ax = plt.subplots(figsize=(13, 7))
        for gap in range(1, max_gap + 1):
            xs = [a + 1 for a in range(n_layers - gap)]
            ys = [means[a, a + gap] for a in range(n_layers - gap)]
            ax.plot(xs, ys, color=gap_cmap(gap - 1), linewidth=1.2)
        ax.set_yscale('log')
        ax.set_xlabel('Encoder split layer  a  (1-indexed)', fontsize=12)
        ax.set_ylabel('Per-element MSE  (log scale)', fontsize=12)
        ax.set_title(
            f'Latent-AR L2 penalty by gap size — {model_type}\n'
            f'{total_pairs:,} position pairs  |  lower = more compatible',
            fontsize=12)
        ax.set_xticks(ticks)
        ax.grid(True, which='both', alpha=0.3)
        sm = cm.ScalarMappable(cmap=gap_cmap,
                               norm=matplotlib.colors.Normalize(vmin=1, vmax=max_gap))
        sm.set_array([])
        plt.colorbar(sm, ax=ax).set_label('Gap  (b − a)', fontsize=11)
        plt.tight_layout()
        plt.savefig(l2_line_path, dpi=150)
        print(f"\nL2 line plot saved to {l2_line_path}")

        # --- L2 heatmap (log scale) ------------------------------------------
        l2_heat_path = f'{plot_prefix}_l2_heatmap.png'
        heat_l2 = np.full((n_layers, n_layers), np.nan)
        for a in range(n_layers):
            for b in range(a + 1, n_layers):
                if pair_counts[a, b] > 0:
                    heat_l2[b, a] = means[a, b]   # row=b, col=a so origin is top-left b=1

        valid = heat_l2[np.isfinite(heat_l2)]
        vmin, vmax = valid.min(), valid.max()
        fig2, ax2 = plt.subplots(figsize=(9, 8))
        im2 = ax2.imshow(heat_l2, origin='upper', aspect='equal',
                         norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma')
        ax2.set_xlabel('Encoder split layer  a  (1-indexed)', fontsize=12)
        ax2.set_ylabel('Latent-AR split layer  b  (1-indexed)', fontsize=12)
        ax2.set_xticks(range(n_layers)); ax2.set_xticklabels(ticks)
        ax2.set_yticks(range(n_layers)); ax2.set_yticklabels(ticks)
        ax2.set_title(
            f'Latent-AR L2 penalty heatmap — {model_type}\n'
            f'{total_pairs:,} position pairs  |  lower = more compatible',
            fontsize=12)
        plt.colorbar(im2, ax=ax2).set_label('Per-element MSE (log scale)', fontsize=11)
        plt.tight_layout()
        plt.savefig(l2_heat_path, dpi=150)
        print(f"L2 heatmap saved to {l2_heat_path}")

        # --- Cosine² line plot -----------------------------------------------
        cos_line_path = f'{plot_prefix}_cosine_lines.png'
        fig3, ax3 = plt.subplots(figsize=(13, 7))
        for gap in range(1, max_gap + 1):
            xs = [a + 1 for a in range(n_layers - gap)]
            ys = [cos_sq_mean[a, a + gap] for a in range(n_layers - gap)]
            ax3.plot(xs, ys, color=gap_cmap(gap - 1), linewidth=1.2)
        ax3.set_xlabel('Encoder split layer  a  (1-indexed)', fontsize=12)
        ax3.set_ylabel('Mean cosine² similarity', fontsize=12)
        ax3.set_title(
            f'Latent-AR cosine² similarity by gap size — {model_type}\n'
            f'{total_pairs:,} position pairs  |  higher = more compatible',
            fontsize=12)
        ax3.set_xticks(ticks)
        ax3.grid(True, alpha=0.3)
        sm3 = cm.ScalarMappable(cmap=gap_cmap,
                                norm=matplotlib.colors.Normalize(vmin=1, vmax=max_gap))
        sm3.set_array([])
        plt.colorbar(sm3, ax=ax3).set_label('Gap  (b − a)', fontsize=11)
        plt.tight_layout()
        plt.savefig(cos_line_path, dpi=150)
        print(f"Cosine² line plot saved to {cos_line_path}")

        # --- Cosine² heatmap -------------------------------------------------
        cos_heat_path = f'{plot_prefix}_cosine_heatmap.png'
        heat_cos = np.full((n_layers, n_layers), np.nan)
        for a in range(n_layers):
            for b in range(a + 1, n_layers):
                if pair_counts[a, b] > 0:
                    heat_cos[b, a] = cos_sq_mean[a, b]

        fig4, ax4 = plt.subplots(figsize=(9, 8))
        im4 = ax4.imshow(heat_cos, origin='upper', aspect='equal',
                         vmin=0, vmax=1, cmap='magma')
        ax4.set_xlabel('Encoder split layer  a  (1-indexed)', fontsize=12)
        ax4.set_ylabel('Latent-AR split layer  b  (1-indexed)', fontsize=12)
        ax4.set_xticks(range(n_layers)); ax4.set_xticklabels(ticks)
        ax4.set_yticks(range(n_layers)); ax4.set_yticklabels(ticks)
        ax4.set_title(
            f'Latent-AR cosine² similarity heatmap — {model_type}\n'
            f'{total_pairs:,} position pairs  |  higher = more compatible',
            fontsize=12)
        plt.colorbar(im4, ax=ax4).set_label('Mean cosine² similarity', fontsize=11)
        plt.tight_layout()
        plt.savefig(cos_heat_path, dpi=150)
        print(f"Cosine² heatmap saved to {cos_heat_path}")

    except ImportError:
        print("\n(install matplotlib to generate a plot)")


def regen():
    """Regenerate plots from all existing checkpoints without running a new scan."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    runs = []

    orig_ckpt = os.path.join(RESULTS_DIR, 'orig_checkpoint.npz')
    if os.path.exists(orig_ckpt):
        runs.append((orig_ckpt, os.path.join(RESULTS_DIR, 'orig')))

    for entry in sorted(os.listdir(RESULTS_DIR)):
        run_dir = os.path.join(RESULTS_DIR, entry)
        if os.path.isdir(run_dir) and entry.startswith('layer_scan_lar_'):
            ckpt = os.path.join(run_dir, 'checkpoint.npz')
            if os.path.exists(ckpt):
                runs.append((ckpt, os.path.join(run_dir, 'lar')))

    if not runs:
        print("No checkpoints found to regenerate.")
        sys.exit(1)

    for ckpt_path, plot_prefix in runs:
        print(f"\nRegenerating plots from {ckpt_path} -> {plot_prefix}_*.png")
        ckpt = np.load(ckpt_path)
        n_layers    = int(np.sqrt(ckpt['penalty_sum'].shape[0] * 2 + 0.25) + 0.5) \
                      if ckpt['penalty_sum'].ndim == 1 \
                      else ckpt['penalty_sum'].shape[0]
        penalty_sum = ckpt['penalty_sum']
        cos_sum     = ckpt['cos_sum'] if 'cos_sum' in ckpt \
                      else np.zeros((n_layers, n_layers), dtype=np.float64)
        pair_counts = ckpt['pair_counts'] if 'pair_counts' in ckpt \
                      else np.full((n_layers, n_layers),
                                   float(ckpt['total_pairs'].item()), dtype=np.float64)
        total_pairs = int(ckpt['total_pairs'].item())
        _report(plot_prefix, penalty_sum, cos_sum, pair_counts, total_pairs, n_layers)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('orig', 'lar', 'alarm', 'regen'):
        print("Usage: python -m projects.latent_ar.scan_layers [orig|lar|alarm|regen]")
        sys.exit(1)
    if sys.argv[1] == 'regen':
        regen()
    else:
        scan(sys.argv[1])
