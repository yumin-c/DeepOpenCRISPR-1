"""
Regenerate SHAP plots for DeepOC from saved CSV.

Usage:
    python plot_shap_dl.py <result_dir>     # specific run
    python plot_shap_dl.py                  # auto-detect latest results/shap_unified_*

    python plot_shap_dl.py results/shap_unified_260403_1234 --top_n 40

Output:
    <result_dir>/shap_unified_signed.jpg
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


COLOR_MAP = {'spacer': '#e74c3c', 'target': '#3498db', 'biofeature': '#2ecc71'}


def plot_unified_signed_shap(csv_path, save_dir, top_n=50):
    df = pd.read_csv(csv_path).sort_values('abs_shap', ascending=False).reset_index(drop=True)
    plot_df = df.head(top_n).iloc[::-1].reset_index(drop=True)  # ascending for barh

    bar_colors = [COLOR_MAP.get(t, '#888888') for t in plot_df['type']]

    fig_h = max(8, top_n * 0.28)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(range(len(plot_df)), plot_df['mean_signed_shap'], color=bar_colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['label'], fontsize=8)
    ax.set_xlabel('Mean signed SHAP  (+ increases activity, − decreases)', fontsize=10)
    ax.set_title(
        f'DeepOC — Top {top_n} features by |mean signed SHAP|\n'
        f'Sequence (per base×position) vs Bio-features, same metric',
        fontsize=11,
    )
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(handles=[
        Patch(facecolor='#e74c3c', label='Spacer (per base × position)'),
        Patch(facecolor='#3498db', label='Target (per base × position)'),
        Patch(facecolor='#2ecc71', label='Bio-feature'),
    ], loc='lower right', fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(save_dir, 'shap_unified_signed.jpg')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def find_latest_result_dir():
    candidates = sorted(glob.glob('results/shap_unified_*'))
    if not candidates:
        raise FileNotFoundError('No results/shap_unified_* directory found.')
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description='Plot SHAP results from saved CSV.')
    parser.add_argument('result_dir', nargs='?', default=None,
                        help='Path to result directory (default: latest results/shap_unified_*)')
    parser.add_argument('--top_n', type=int, default=50,
                        help='Number of top features to show (default: 50)')
    args = parser.parse_args()

    result_dir = args.result_dir or find_latest_result_dir()
    csv_path = os.path.join(result_dir, 'shap_unified_signed.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    print(f'Reading: {csv_path}')
    plot_unified_signed_shap(csv_path, result_dir, top_n=args.top_n)


if __name__ == '__main__':
    main()
