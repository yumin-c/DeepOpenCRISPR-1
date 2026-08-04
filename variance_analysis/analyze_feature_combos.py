"""
Residual / variance-explained comparison: the 7 combinations of the three
handcrafted feature blocks (GC, positional, PAM) versus DeepOC alone, in each of
the three matched (on-target) strata. Out-of-fold R² with 95% bootstrap CI.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from feature_ceiling_report import (load, feature_blocks, oof, r2, boot_r2,
                                     MODELS, FOLDS, ACT, SUBSETS)

B, SEED = 2000, 0
OUT = os.path.dirname(os.path.abspath(__file__))

# 7 combinations of the 3 feature blocks, plus DeepOC alone
COMBOS = {
    'GC': ['GC'],
    'positional': ['positional'],
    'PAM': ['PAM'],
    'GC+positional': ['GC', 'positional'],
    'GC+PAM': ['GC', 'PAM'],
    'positional+PAM': ['positional', 'PAM'],
    'GC+positional+PAM': ['GC', 'positional', 'PAM'],
}
ORDER = list(COMBOS) + ['DeepOC']
STRATA = [('matched_NGG', 'Matched, NGG'),
          ('matched_nonNGG', 'Matched, non-NGG'),
          ('matched_all', 'Matched, all PAM')]

df = load()
draws_rows, summ_rows = [], []
for key, label in STRATA:
    mask = SUBSETS[key][1]
    sub = df[mask(df)]
    sub = sub[sub['Fold'].isin(FOLDS)].reset_index(drop=True)
    y = sub[ACT].values.astype(float)
    blocks = feature_blocks(sub)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(sub), size=(B, len(sub)))
    print(f'[{key}] n={len(sub)}', flush=True)

    preds = {}
    for name, keys in COMBOS.items():
        X = np.hstack([blocks[k] for k in keys])
        preds[name] = oof(sub, X, y, MODELS['GBM'])
    preds['DeepOC'] = sub['prediction'].values.astype(float)   # raw OOF

    for name in ORDER:
        p = preds[name]
        bvals = boot_r2(y, p, idx)
        lo, hi = np.percentile(bvals, [2.5, 97.5])
        summ_rows.append(dict(stratum=key, stratum_label=label, feature_set=name,
                              r2=r2(y, p), ci_lo=lo, ci_hi=hi))
        for v in bvals:
            draws_rows.append(dict(stratum=label, feature_set=name, r2=v))

summ = pd.DataFrame(summ_rows)
summ.to_csv(f'{OUT}/feature_combos_vs_deepoc.csv', index=False)
draws = pd.DataFrame(draws_rows)

print('\n=== point R² ===')
print(summ.pivot(index='feature_set', columns='stratum', values='r2').reindex(ORDER).round(3).to_string())

# ---- seaborn bar plot, one panel per stratum ----
sns.set_theme(style='ticks', context='talk')
palette = {name: '#5b8fb0' for name in COMBOS}
palette['DeepOC'] = '#e6550d'

fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
for ax, (key, label) in zip(axes, STRATA):
    sub = draws[draws.stratum == label]
    sns.barplot(data=sub, x='feature_set', y='r2', order=ORDER, hue='feature_set',
                hue_order=ORDER, palette=palette, legend=False,
                errorbar=('pi', 95), err_kws={'linewidth': 1.2}, capsize=0.15, ax=ax)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xlabel('')
    ax.set_ylabel('Out-of-fold R²' if ax is axes[0] else '')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
    ax.text(0.03, 0.97, label, transform=ax.transAxes, va='top', ha='left',
            fontsize=14, fontweight='bold')
    sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(f'{OUT}/feature_combos_vs_deepoc.png', dpi=150)
print(f'\nwrote {OUT}/feature_combos_vs_deepoc.png')
