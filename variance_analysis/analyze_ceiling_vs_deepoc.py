"""
How much of the reproducible activity variance is captured by DeepOC vs by the
handcrafted sequence features?

Uses the R1/R2 replicate measurements (OpenCRISPR-1_replicates.csv) to establish a
reproducibility ceiling -- the maximum R^2 any sequence-based model could reach,
since the target carries measurement noise -- and compares against it:
  - GC content
  - position x nucleotide one-hot
  - GC + one-hot
  - DeepOC (sequence-only ablation model) out-of-fold prediction
  - DeepOC + GC + one-hot   (does adding features to DeepOC help?)

Reviewer context:
  Point 1 -- is the CNN's edge architecture or the engineered features? If
             DeepOC > (GC + one-hot) and DeepOC + features == DeepOC, the edge
             is architectural and the features are subsumed by the sequence.
  Point 6 -- how much per-target variance do the features explain, and is the
             unexplained part noise? Compare each model's R^2 to the ceiling.

Target y = "OpenCRISPR-1 activity (day 7, %)" (what DeepOC was trained on).
Noise ceiling is estimated from R1 vs R2 and reported three ways:
  rho    = corr(R1,R2)            reliability of a single measurement
                                  (= max R^2 for a single-replicate target)
  rho^2                           R^2 of one experiment predicting another
                                  (most conservative, assumption-light)
  SB     = 2*rho/(1+rho)          reliability of a 2-replicate mean
                                  (= max R^2 if y is the mean of two replicates)
Since `activity` ~ mean of two replicates (corr 0.998), the ceiling band is
[rho^2, SB], with rho a middle reference.

Evaluation: 5-fold out-of-fold on the Fold0..4 rows, folds taken from the
dataset's own Fold column so DeepOC's OOF predictions stay leakage-free and the
meta-model's DeepOC feature is always out-of-fold.
"""

import os
# shared many-core box: cap BLAS/OMP threads so RF's own parallelism does not
# oversubscribe (this was making the run pathologically slow).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, 'data', 'OpenCRISPR-1_replicates.csv')
ABL = os.path.join(_ROOT, 'results', 'deepoc')   # final DeepOC (PAM-branch) predictions
ACT = 'OpenCRISPR-1 activity (day 7, %)'
NTS = 'ACGT'
SEED = 0
FOLDS = [f'Fold{i}' for i in range(5)]
RNG = np.random.default_rng(SEED)


# ---------------------------------------------------------------- data / features
def load():
    d = pd.read_csv(DATA)
    cv = pd.read_csv(f'{ABL}/cv_predictions.csv')
    te = pd.read_csv(f'{ABL}/test_predictions.csv')
    pred = pd.concat([cv, te])[['Spacer', 'Target', 'prediction']]
    d = d.merge(pred, on=['Spacer', 'Target'], how='left')
    assert d['prediction'].notna().all(), 'missing DeepOC predictions'
    return d


def gc(seq):
    return np.array([(s.count('G') + s.count('C')) / len(s) for s in seq], np.float32)


def target_onehot(seqs):
    L = len(seqs.iloc[0])
    X = np.zeros((len(seqs), L * 4), np.float32)
    for i, s in enumerate(seqs):
        for p, ch in enumerate(s):
            k = NTS.find(ch)
            if k >= 0:
                X[i, p * 4 + k] = 1.0
    return X


def feature_blocks(df):
    return {
        'GC': np.column_stack([gc(df.Spacer), gc(df.Target)]),
        'onehot': target_onehot(df.Target),
        'deepoc': df[['prediction']].values.astype(np.float32),
    }


# ---------------------------------------------------------------- evaluation
def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def oof(df, X, y, model_fn):
    """Out-of-fold predictions using the dataset's own Fold column."""
    pred = np.full(len(y), np.nan)
    fold = df['Fold'].values
    for f in FOLDS:
        te = fold == f
        tr = ~te
        m = model_fn()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def boot_r2_ci(y, p, n=300):
    idx = np.arange(len(y))
    vals = [r2(y[b], p[b]) for b in (RNG.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- ceiling
def ceiling(df):
    r1, r2_ = df['Replicate 1'].values, df['Replicate 2'].values
    rho = pearsonr(r1, r2_)[0]
    # bootstrap CI on rho
    idx = np.arange(len(df))
    rhos = [pearsonr(r1[b], r2_[b])[0] for b in (RNG.choice(idx, len(idx)) for _ in range(300))]
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return dict(rho=rho, rho2=rho ** 2, SB=2 * rho / (1 + rho),
                rho_ci=(lo, hi))


# ---------------------------------------------------------------- run one subset
MODELS = {
    'RF':  lambda: RandomForestRegressor(n_estimators=200, n_jobs=8, random_state=SEED),
    'GBM': lambda: HistGradientBoostingRegressor(random_state=SEED),
    'lin': lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))),
}

FEATURE_SETS = {
    'GC':                 ['GC'],
    'onehot':             ['onehot'],
    'GC+onehot':          ['GC', 'onehot'],
    'DeepOC':             ['deepoc'],
    'DeepOC+GC+onehot':   ['deepoc', 'GC', 'onehot'],
}


def run_subset(df, label):
    sub = df[df['Fold'].isin(FOLDS)].copy()          # OOF-evaluable rows only
    y = sub[ACT].values.astype(float)
    blocks = feature_blocks(sub)

    c = ceiling(sub)
    print(f'\n{"="*72}\n{label}   (n={len(sub)})')
    print(f'reproducibility ceiling: rho={c["rho"]:.3f} '
          f'(95% CI {c["rho_ci"][0]:.3f}-{c["rho_ci"][1]:.3f})  '
          f'rho^2={c["rho2"]:.3f}  SB(mean-of-2)={c["SB"]:.3f}')
    print(f'ceiling band for R^2: [{c["rho2"]:.3f}, {c["SB"]:.3f}]')

    # DeepOC alone is already an OOF prediction -> direct R^2 (no refit)
    dp = sub['prediction'].values.astype(float)
    lo, hi = boot_r2_ci(y, dp)
    print(f'\n{"model":<20}{"R^2 (RF)":>10}{"R^2 (GBM)":>11}{"R^2 (lin)":>11}')
    print(f'{"DeepOC (raw OOF)":<20}{r2(y, dp):>10.3f}{"":>11}{"":>11}   '
          f'[{lo:.3f}, {hi:.3f}]')

    results = {'DeepOC (raw OOF)': r2(y, dp)}
    sys.stdout.flush()
    for name, keys in FEATURE_SETS.items():
        X = np.hstack([blocks[k] for k in keys])
        row = {}
        for mtag, mfn in MODELS.items():
            row[mtag] = r2(y, oof(sub, X, y, mfn))
        results[name] = row
        print(f'{name:<20}{row["RF"]:>10.3f}{row["GBM"]:>11.3f}{row["lin"]:>11.3f}')
        sys.stdout.flush()
    return c, results


def main():
    df = load()
    subsets = {
        'Matched NGG PAM (on-target, primary)': df.Info == 'Matched target with NGG PAM',
        'All matched (on-target)': df.Info.str.startswith('Matched'),
        'Full dataset': pd.Series(True, index=df.index),
    }
    for label, mask in subsets.items():
        run_subset(df[mask], label)


if __name__ == '__main__':
    main()
