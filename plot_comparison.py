import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, pearsonr

# ── Load data ──
ML_DIR = 'results/ml_260920_1223'

# The ML baselines are one-hot only, matching sequence-only DeepOC.
MODES = [
    ('onehot_only', 'OneHot'),
]

ml_modes = {label: pd.read_csv(f'{ML_DIR}/cv_per_fold_{name}.csv') for name, label in MODES}

# ── Prepare DL data ──
# results/deepoc/cv_metrics.csv is aggregated by stratum, so the per-fold
# numbers are recomputed from the held-out CV predictions instead.
dl_pred = pd.read_csv('results/deepoc/cv_predictions.csv')
ACTIVITY = 'OpenCRISPR-1 activity (day 7, %)'
dl_rows = []
for fold_idx in range(5):
    f = dl_pred[dl_pred['Fold'] == f'Fold{fold_idx}']
    dl_rows.append({
        'model': 'DL (DeepOC)',
        'fold': fold_idx,
        'spearman': spearmanr(f[ACTIVITY], f['prediction']).statistic,
        'pearson': pearsonr(f[ACTIVITY], f['prediction']).statistic,
    })
dl_df = pd.DataFrame(dl_rows)

# ── Best ML mode per model: pick whichever mode gave the highest mean Spearman ──
ml_best_rows = []
for model_name in next(iter(ml_modes.values()))['model'].unique():
    best_label = max(
        ml_modes,
        key=lambda label: ml_modes[label].loc[ml_modes[label]['model'] == model_name, 'spearman'].mean(),
    )
    df = ml_modes[best_label]
    subset = df[df['model'] == model_name].copy()
    subset['input_mode'] = best_label
    ml_best_rows.append(subset)
ml_best = pd.concat(ml_best_rows)

# ── Sort ML models by mean Spearman (descending) ──
ml_order = (ml_best.groupby('model')['spearman'].mean()
            .sort_values(ascending=False).index.tolist())

# ── Combined order: DL first, then ML ──
all_models = ['DL (DeepOC)'] + ml_order
combined = pd.concat([dl_df, ml_best], ignore_index=True)

# ── Labels with input mode ──
xlabels = []
for m in all_models:
    if m.startswith('DL'):
        xlabels.append(m)
    else:
        mode = ml_best[ml_best['model'] == m]['input_mode'].iloc[0]
        xlabels.append(f'{m}\n({mode})')

# ── Colors ──
dl_color = '#E74C3C'
ml_color = '#3498DB'
colors = [dl_color if m.startswith('DL') else ml_color for m in all_models]
positions = list(range(len(all_models)))

# ── Plot 1: Spearman box plot (vertical) ──
fig, ax = plt.subplots(figsize=(14, 7))

box_data = [combined[combined['model'] == m]['spearman'].values for m in all_models]

bp = ax.boxplot(box_data, positions=positions, vert=True, widths=0.6,
                patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=4))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for i, m in enumerate(all_models):
    vals = combined[combined['model'] == m]['spearman'].values
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
    ax.scatter([i] * len(vals) + jitter, vals, color='black', s=15, zorder=5, alpha=0.6)

ax.set_xticks(positions)
ax.set_xticklabels(xlabels, fontsize=9, rotation=30, ha='right')
ax.set_ylabel('Spearman Correlation (5-fold CV)', fontsize=11)
ax.set_title('Model Performance Comparison — Spearman', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

dl_patch = mpatches.Patch(color=dl_color, alpha=0.7, label='Deep Learning')
ml_patch = mpatches.Patch(color=ml_color, alpha=0.7, label='Machine Learning')
ax.legend(handles=[dl_patch, ml_patch], loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('results/model_comparison_spearman.jpg', dpi=300, bbox_inches='tight')
plt.close()

# ── Plot 2: Pearson box plot (vertical) ──
fig, ax = plt.subplots(figsize=(14, 7))

box_data_p = [combined[combined['model'] == m]['pearson'].values for m in all_models]

bp = ax.boxplot(box_data_p, positions=positions, vert=True, widths=0.6,
                patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=4))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for i, m in enumerate(all_models):
    vals = combined[combined['model'] == m]['pearson'].values
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
    ax.scatter([i] * len(vals) + jitter, vals, color='black', s=15, zorder=5, alpha=0.6)

ax.set_xticks(positions)
ax.set_xticklabels(xlabels, fontsize=9, rotation=30, ha='right')
ax.set_ylabel('Pearson Correlation (5-fold CV)', fontsize=11)
ax.set_title('Model Performance Comparison — Pearson', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax.legend(handles=[dl_patch, ml_patch], loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('results/model_comparison_pearson.jpg', dpi=300, bbox_inches='tight')
plt.close()

# ── Plot 3: Combined (Spearman + Pearson stacked) ──
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

for ax, metric, title in [(axes[0], 'spearman', 'Spearman'), (axes[1], 'pearson', 'Pearson')]:
    box_data = [combined[combined['model'] == m][metric].values for m in all_models]

    bp = ax.boxplot(box_data, positions=positions, vert=True, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markersize=4))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, m in enumerate(all_models):
        vals = combined[combined['model'] == m][metric].values
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter([i] * len(vals) + jitter, vals, color='black', s=15, zorder=5, alpha=0.6)

    ax.set_ylabel(f'{title} Correlation', fontsize=11)
    ax.set_title(f'{title}', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

axes[1].set_xticks(positions)
axes[1].set_xticklabels(xlabels, fontsize=9, rotation=30, ha='right')
axes[0].legend(handles=[dl_patch, ml_patch], loc='upper right', fontsize=10)

plt.suptitle('5-Fold CV Model Performance Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/model_comparison_combined.jpg', dpi=300, bbox_inches='tight')
plt.close()

print('Saved:')
print('  results/model_comparison_spearman.jpg')
print('  results/model_comparison_pearson.jpg')
print('  results/model_comparison_combined.jpg')
