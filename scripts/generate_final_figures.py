"""Generate final publication-quality figures for the paper.

Creates:
1. Main results table comparison (bar chart)
2. Error over horizon comparison (line plot)
3. Per-seed consistency plot
"""

import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ABLATION_DIR = '/mnt/hdd12t/outputs/scenario_dreamer_ablation_v3'
FIGURE_DIR = '/mnt/hdd12t/outputs/scenario_dreamer_figures'
SEEDS = [42, 123, 7, 0, 99]

os.makedirs(FIGURE_DIR, exist_ok=True)

# CV/CA baselines from evaluate_baselines.py
CV_BASELINES = {
    'Constant Velocity': {
        'ade_1s': 0.246, 'fde_1s': 0.494,
        'ade_2s': 0.597, 'fde_2s': 1.377,
        'ade_3s': 1.108, 'fde_3s': 2.929,
    },
}


def get_best_metrics(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r.get('val/ade') and r['val/ade']]
    if not val_rows:
        return None
    best = min(val_rows, key=lambda r: float(r['val/ade']))
    return {k: float(best[k]) for k in best if k.startswith('val/') and best[k]}


def load_all_results():
    models = ['baseline', 'lane_conditioned', 'dual_supervised']
    all_results = {}
    for model in models:
        results = []
        for seed in SEEDS:
            path = f'{ABLATION_DIR}/{model}_seed{seed}/csv_logs/version_0/metrics.csv'
            try:
                m = get_best_metrics(path)
                if m:
                    results.append(m)
            except:
                pass
        all_results[model] = results
    return all_results


def plot_main_comparison(all_results):
    """Bar chart comparing all methods at 1s, 2s, 3s horizons."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    horizons = [('1s', 'val/ade_1s', 'val/fde_1s'),
                ('2s', 'val/ade_2s', 'val/fde_2s'),
                ('3s', 'val/ade_3s', 'val/fde_3s')]

    model_labels = {
        'baseline': 'LSTM\nBaseline',
        'lane_conditioned': 'Lane-\nConditioned',
        'dual_supervised': 'Dual\nSupervised',
    }

    colors = {
        'cv': '#888888',
        'baseline': '#4ECDC4',
        'lane_conditioned': '#FF6B6B',
        'dual_supervised': '#45B7D1',
    }

    for ax_idx, (horizon, ade_key, fde_key) in enumerate(horizons):
        methods = ['cv', 'baseline', 'lane_conditioned', 'dual_supervised']
        ade_means = []
        ade_stds = []
        labels = []

        # CV baseline
        cv_ade_key = f'ade_{horizon}'
        ade_means.append(CV_BASELINES['Constant Velocity'][cv_ade_key])
        ade_stds.append(0)
        labels.append('Constant\nVelocity')

        # Learned models
        for model in ['baseline', 'lane_conditioned', 'dual_supervised']:
            results = all_results[model]
            vals = [r.get(ade_key, r.get('val/ade', 0)) for r in results]
            ade_means.append(np.mean(vals))
            ade_stds.append(np.std(vals))
            labels.append(model_labels[model])

        x = np.arange(len(methods))
        bars = ax_idx == 0  # only show y-label on first

        bar_colors = [colors[m] for m in methods]
        ax = axes[ax_idx]
        bars = ax.bar(x, ade_means, yerr=ade_stds, capsize=5,
                      color=bar_colors, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f'{horizon} Horizon', fontsize=13, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel('ADE (m)', fontsize=12)
        ax.set_ylim(0, max(ade_means) * 1.3)

        # Add value labels
        for bar, mean in zip(bars, ade_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'main_comparison_v3.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_error_over_horizon(all_results):
    """Line plot showing error accumulation over prediction horizon."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # For this, we need per-step errors from the CSV — we'll approximate from horizon metrics
    horizons_t = [1, 2, 3]  # seconds

    colors = {
        'cv': '#888888',
        'baseline': '#4ECDC4',
        'lane_conditioned': '#FF6B6B',
        'dual_supervised': '#45B7D1',
    }

    # ADE plot
    for model, label, marker in [
        ('cv', 'Constant Velocity', 's'),
        ('baseline', 'LSTM Baseline', 'o'),
        ('lane_conditioned', 'Lane-Conditioned', '^'),
        ('dual_supervised', 'Dual Supervised', 'D'),
    ]:
        if model == 'cv':
            ade_vals = [0.246, 0.597, 1.108]
            fde_vals = [0.494, 1.377, 2.929]
            ax1.plot(horizons_t, ade_vals, color=colors[model], marker=marker,
                     label=label, linewidth=2, markersize=8, linestyle='--')
            ax2.plot(horizons_t, fde_vals, color=colors[model], marker=marker,
                     label=label, linewidth=2, markersize=8, linestyle='--')
        else:
            results = all_results[model]
            ade_1s = np.mean([r.get('val/ade_1s', 0) for r in results])
            ade_2s = np.mean([r.get('val/ade_2s', 0) for r in results])
            ade_3s = np.mean([r.get('val/ade_3s', r.get('val/ade', 0)) for r in results])
            fde_1s = np.mean([r.get('val/fde_1s', 0) for r in results])
            fde_2s = np.mean([r.get('val/fde_2s', 0) for r in results])
            fde_3s = np.mean([r.get('val/fde_3s', r.get('val/fde', 0)) for r in results])

            ax1.plot(horizons_t, [ade_1s, ade_2s, ade_3s], color=colors[model],
                     marker=marker, label=label, linewidth=2, markersize=8)
            ax2.plot(horizons_t, [fde_1s, fde_2s, fde_3s], color=colors[model],
                     marker=marker, label=label, linewidth=2, markersize=8)

    for ax, metric in [(ax1, 'ADE'), (ax2, 'FDE')]:
        ax.set_xlabel('Prediction Horizon (s)', fontsize=12)
        ax.set_ylabel(f'{metric} (m)', fontsize=12)
        ax.set_title(f'{metric} over Prediction Horizon', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xticks([1, 2, 3])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'error_over_horizon_v3.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_per_seed_consistency(all_results):
    """Scatter plot showing per-seed ADE@3s for each model."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        'baseline': '#4ECDC4',
        'lane_conditioned': '#FF6B6B',
        'dual_supervised': '#45B7D1',
    }
    labels = {
        'baseline': 'LSTM Baseline',
        'lane_conditioned': 'Lane-Conditioned',
        'dual_supervised': 'Dual Supervised',
    }

    x_offsets = {'baseline': -0.15, 'lane_conditioned': 0.0, 'dual_supervised': 0.15}

    for model in ['baseline', 'lane_conditioned', 'dual_supervised']:
        results = all_results[model]
        vals = [r.get('val/ade_3s', r.get('val/ade', 0)) for r in results]
        x = np.arange(len(vals)) + x_offsets[model]
        ax.scatter(x, vals, c=colors[model], label=labels[model],
                   s=100, zorder=3, edgecolors='black', linewidth=0.5)
        # Mean line
        ax.axhline(np.mean(vals), color=colors[model], linestyle='--',
                    alpha=0.5, linewidth=1)

    # CV baseline
    ax.axhline(1.108, color='#888888', linestyle=':', linewidth=2,
               label='Constant Velocity (1.108m)')

    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f'Seed {s}' for s in SEEDS])
    ax.set_ylabel('ADE@3s (m)', fontsize=12)
    ax.set_title('Per-Seed Consistency (ADE@3s)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'per_seed_consistency_v3.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_decoder_comparison():
    """Compare LSTM vs MLP decoder (v2 vs v3) for paper."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # V2 results (LSTM decoder)
    v2_means = {'baseline': 1.213, 'lane_conditioned': 1.184, 'dual_supervised': 1.163}
    v2_stds = {'baseline': 0.068, 'lane_conditioned': 0.090, 'dual_supervised': 0.101}

    # V3 results (MLP decoder)
    v3_means = {'baseline': 1.096, 'lane_conditioned': 1.086, 'dual_supervised': 1.089}
    v3_stds = {'baseline': 0.066, 'lane_conditioned': 0.068, 'dual_supervised': 0.069}

    models = ['baseline', 'lane_conditioned', 'dual_supervised']
    labels = ['LSTM Baseline', 'Lane-Conditioned', 'Dual Supervised']
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, [v2_means[m] for m in models],
                   width, yerr=[v2_stds[m] for m in models],
                   label='LSTM Decoder (v2)', color='#FF6B6B', edgecolor='black',
                   linewidth=0.5, capsize=5)
    bars2 = ax.bar(x + width/2, [v3_means[m] for m in models],
                   width, yerr=[v3_stds[m] for m in models],
                   label='MLP Decoder (v3)', color='#45B7D1', edgecolor='black',
                   linewidth=0.5, capsize=5)

    # CV line
    ax.axhline(1.108, color='#888888', linestyle=':', linewidth=2,
               label='Constant Velocity')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('ADE@3s (m)', fontsize=12)
    ax.set_title('LSTM vs MLP Decoder: ADE@3s Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0.95, 1.35)
    ax.grid(True, alpha=0.3, axis='y')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, 'decoder_comparison_v2_v3.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def main():
    print("Loading results...")
    all_results = load_all_results()

    for model, results in all_results.items():
        print(f"  {model}: {len(results)} seeds loaded")

    print("\nGenerating figures...")
    plot_main_comparison(all_results)
    plot_error_over_horizon(all_results)
    plot_per_seed_consistency(all_results)
    plot_decoder_comparison()

    print("\nAll figures saved to:", FIGURE_DIR)


if __name__ == '__main__':
    main()
