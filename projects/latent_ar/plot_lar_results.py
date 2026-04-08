"""
Visualises the output of lar_test.py.

Loads the most recent JSON for each metric from lar_results/ and produces
three figures:

  1. Rollout Fidelity      — KL(AR||LAR) and top-1 agreement vs rollout step
  2. Latent Traj. Diverge  — L2 distance and L2/magnitude vs rollout step
  3. Latent Rollout CE     — LAR CE, baseline CE, and their delta vs rollout step

Run from the repo root:
    python -m projects.latent_ar.plot_lar_results
"""

import glob
import json
import os

import time

import matplotlib.pyplot as plt
import numpy as np

_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_DIR, 'lar_results')


def latest(prefix):
    pattern = os.path.join(RESULTS_DIR, f'{prefix}_*.json')
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results found matching {pattern}")
    path = files[-1]
    print(f"Loading {os.path.basename(path)}")
    with open(path) as f:
        return json.load(f)


def smooth(arr, window=20):
    """Simple moving average for readability on 896-step curves."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def plot_rollout_fidelity(ax_kl, ax_agree, data):
    steps  = np.arange(len(data['per_step']['kl']))
    kl     = np.array(data['per_step']['kl'])
    agree  = np.array(data['per_step']['agree'])

    ax_kl.plot(steps, kl, alpha=0.25, color='steelblue', linewidth=0.8)
    ax_kl.plot(steps, smooth(kl), color='steelblue', linewidth=1.8, label='KL(AR||LAR)')
    ax_kl.set_ylabel('KL divergence')
    ax_kl.set_title('Rollout Fidelity — KL(AR||LAR)')
    ax_kl.legend()

    ax_agree.plot(steps, agree, alpha=0.25, color='darkorange', linewidth=0.8)
    ax_agree.plot(steps, smooth(agree), color='darkorange', linewidth=1.8, label='Top-1 agreement')
    ax_agree.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax_agree.set_ylabel('Agreement rate')
    ax_agree.set_title('Rollout Fidelity — Top-1 Agreement')
    ax_agree.legend()


def plot_trajectory_divergence(ax_l2, ax_norms, data):
    steps = np.arange(len(data['per_step']['l2']))
    l2    = np.array(data['per_step']['l2'])
    mag   = data['summary']['mean_magnitude']

    ax_l2.plot(steps, l2, alpha=0.25, color='firebrick', linewidth=0.8)
    ax_l2.plot(steps, smooth(l2), color='firebrick', linewidth=1.8, label='L2 distance')
    ax_l2.axhline(mag, color='grey', linestyle='--', linewidth=1.0, label=f'Mean h_a magnitude ({mag:.1f})')
    ax_l2.set_ylabel('L2 distance')
    ax_l2.set_title('Latent Trajectory Divergence — L2')
    ax_l2.legend()

    pred_norm = np.array(data['per_step']['pred_norm'])
    gt_norm   = np.array(data['per_step']['gt_norm'])
    ax_norms.plot(steps, pred_norm, alpha=0.25, color='firebrick', linewidth=0.8)
    ax_norms.plot(steps, smooth(pred_norm), color='firebrick', linewidth=1.8, label='Predicted (h_b) norm')
    ax_norms.plot(steps, gt_norm, alpha=0.25, color='steelblue', linewidth=0.8)
    ax_norms.plot(steps, smooth(gt_norm), color='steelblue', linewidth=1.8, label='Ground truth (h_a) norm')
    ax_norms.set_ylabel('Vector norm')
    ax_norms.set_title('Latent Trajectory — Vector Norms')
    ax_norms.legend()


def plot_rollout_ce(ax, data):
    steps    = np.arange(len(data['per_step']['ce']))
    ce       = np.array(data['per_step']['ce'])
    base_ce  = np.array(data['per_step']['base_ce'])
    delta    = ce - base_ce

    ax.plot(steps, base_ce, alpha=0.2, color='steelblue', linewidth=0.8)
    ax.plot(steps, smooth(base_ce), color='steelblue', linewidth=1.8, label='Baseline CE (AR)')
    ax.plot(steps, ce, alpha=0.2, color='firebrick', linewidth=0.8)
    ax.plot(steps, smooth(ce), color='firebrick', linewidth=1.8, label='LAR rollout CE')

    ax2 = ax.twinx()
    ax2.plot(steps, delta, alpha=0.15, color='darkorange', linewidth=0.8)
    ax2.plot(steps, smooth(delta), color='darkorange', linewidth=1.8, linestyle='--', label='Delta (LAR − base)')
    ax2.axhline(0, color='darkorange', linestyle=':', linewidth=0.8)
    ax2.set_ylabel('CE delta', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    ax.set_ylabel('Cross-entropy')
    ax.set_title('Latent Rollout CE')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)


def run():
    rf  = latest('rollout_fidelity')
    ltd = latest('latent_trajectory_divergence')
    lrc = latest('latent_rollout_ce')

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('LAR Test Results', fontsize=14, fontweight='bold')

    plot_rollout_fidelity(axes[0, 0], axes[1, 0], rf)
    plot_trajectory_divergence(axes[0, 1], axes[1, 1], ltd)
    plot_rollout_ce(axes[0, 2], lrc)
    axes[1, 2].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel('Rollout step')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(RESULTS_DIR, f'lar_results_{timestamp}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    run()
