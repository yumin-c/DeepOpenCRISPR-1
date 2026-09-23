"""
Generate figure-ready outputs for the feature-vs-DeepOC vs reproducibility-ceiling
analysis (reviewer points 1 and 6).

Produces, under results/feature_variance_ceiling_260725/:
  model_performance.csv        point R^2 + bootstrap 95% CI, every model x regressor x subset
  reproducibility_ceiling.csv  replicate-based ceiling (reliability rho, and rho^2) + CI
  paired_deltas.csv            paired-bootstrap DeltaR^2 for the key comparisons
  bootstrap_draws.csv          full bootstrap R^2 draws (long format) for custom error bars
  oof_predictions_<subset>.csv per-row truth / DeepOC / feature-model / combined predictions
  README.txt                   column dictionary

All models are evaluated by out-of-fold prediction on the dataset's own
Fold0..4 assignment; uncertainty is a paired nonparametric bootstrap over rows
(same resample indices across models within a subset, enabling DeltaR^2 CIs).
"""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

from analyze_ceiling_vs_deepoc import load, oof, r2, MODELS, FOLDS, ACT

OUTDIR = os.path.dirname(os.path.abspath(__file__))
B = 2000
SEED = 0
NTS = 'ACGT'

# Target layout (30 nt): 5' context idx 0-4 | protospacer idx 4-23 (manuscript
# positions 20..1) | PAM idx 24-27 | 3' context idx 28-29.
PROTO_IDX = list(range(4, 24))
PAM_IDX = list(range(24, 28))

SUBSETS = {
    'matched_NGG':     ('Matched, NGG PAM (on-target)',
                        lambda d: d.Info == 'Matched target with NGG PAM'),
    'matched_nonNGG':  ('Matched, non-NGG PAM (on-target)',
                        lambda d: d.Info == 'Matched target with non-NGG PAM'),
    'matched_all':     ('Matched, all PAM (on-target)',
                        lambda d: d.Info.str.startswith('Matched')),
}

# The handcrafted features characterized in the manuscript: GC content,
# position-specific nucleotide composition (protospacer), and PAM.
FEATURE_SETS = {
    'GC':              ['GC'],
    'positional':      ['positional'],
    'PAM':             ['PAM'],
    'features':        ['GC', 'positional', 'PAM'],   # all handcrafted features
    'DeepOC':          ['deepoc'],
    'DeepOC+features': ['deepoc', 'GC', 'positional', 'PAM'],
}

# regressor treated as the headline for each feature set. DeepOC (the final
# PAM-branch model) is used as its own raw out-of-fold prediction -- it is
# well-calibrated across all strata, including the low-activity non-NGG stratum,
# so no post-hoc regressor is needed. Feature blocks use GBM.
PRIMARY = {'GC': 'GBM', 'positional': 'GBM', 'PAM': 'GBM', 'features': 'GBM',
           'DeepOC': 'raw', 'DeepOC+features': 'GBM'}


def gc_frac(seqs):
    return np.array([(s.count('G') + s.count('C')) / len(s) for s in seqs], np.float32)


def onehot_idx(seqs, idxs):
    X = np.zeros((len(seqs), len(idxs) * 4), np.float32)
    for i, s in enumerate(seqs):
        for j, p in enumerate(idxs):
            k = NTS.find(s[p])
            if k >= 0:
                X[i, j * 4 + k] = 1.0
    return X


def pam_block(seqs):
    """4-position PAM one-hot (position x nucleotide; no derived class features)."""
    return onehot_idx(seqs, PAM_IDX)


def feature_blocks(df):
    spc, tgt = df['Spacer'].values, df['Target'].values
    GC = np.column_stack([gc_frac(spc), gc_frac(tgt)])   # raw GC content only
    return {
        'GC': GC,
        'positional': onehot_idx(tgt, PROTO_IDX),
        'PAM': pam_block(tgt),
        'deepoc': df[['prediction']].values.astype(np.float32),
    }


def boot_r2(y, p, idx):
    """Bootstrap R^2 over precomputed resample index matrix idx (B x n)."""
    yb, pb = y[idx], p[idx]
    ss_res = ((yb - pb) ** 2).sum(1)
    ss_tot = ((yb - yb.mean(1, keepdims=True)) ** 2).sum(1)
    return 1.0 - ss_res / ss_tot


def boot_corr(a, b, idx):
    ab, bb = a[idx], b[idx]
    am = ab - ab.mean(1, keepdims=True)
    bm = bb - bb.mean(1, keepdims=True)
    num = (am * bm).sum(1)
    den = np.sqrt((am ** 2).sum(1) * (bm ** 2).sum(1))
    return num / den


def ci(v):
    return np.percentile(v, [2.5, 97.5])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = load()

    perf_rows, ceil_rows, delta_rows, draw_rows = [], [], [], []

    for key, (label, mask) in SUBSETS.items():
        sub = df[mask(df)]
        sub = sub[sub['Fold'].isin(FOLDS)].reset_index(drop=True)
        n = len(sub)
        y = sub[ACT].values.astype(float)
        blocks = feature_blocks(sub)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(B, n))          # shared resamples for this subset
        print(f'[{key}] n={n}  bootstrapping B={B}', flush=True)

        # ---- reproducibility ceiling ----
        # The target (`activity`) is a single background-subtracted value, not the
        # mean of the two replicates, so the Spearman-Brown (2-replicate-mean)
        # bound does not apply. The ceiling band is [rho^2, rho]:
        #   rho   = reliability of a single measurement = max R^2 for a
        #           single-measurement target (upper bound)
        #   rho^2 = R^2 of one experiment predicting an independent repeat
        #           (both endpoints noisy; conservative lower bound)
        r1, r2m = sub['Replicate 1'].values.astype(float), sub['Replicate 2'].values.astype(float)
        rho = np.corrcoef(r1, r2m)[0, 1]
        rho_b = boot_corr(r1, r2m, idx)
        rho2_b = rho_b ** 2
        ceil_rows.append(dict(
            subset=key, subset_label=label, n=n,
            rho=rho, rho_lo=ci(rho_b)[0], rho_hi=ci(rho_b)[1],
            r2_ceiling=rho,                       # reliability (upper bound)
            r2_ceiling_lo=ci(rho_b)[0], r2_ceiling_hi=ci(rho_b)[1],
            r2_ceiling_conservative=rho ** 2,     # one experiment predicting another
            r2_ceiling_conservative_lo=ci(rho2_b)[0], r2_ceiling_conservative_hi=ci(rho2_b)[1]))

        # ---- models ----
        oof_pred = {}   # (feature_set, regressor) -> oof prediction vector
        # DeepOC alone: its own out-of-fold prediction, no refit
        dp = sub['prediction'].values.astype(float)
        oof_pred[('DeepOC', 'raw')] = dp

        for fs, keys in FEATURE_SETS.items():
            if fs == 'DeepOC':
                continue  # handled as raw above; also fit for completeness below? no
            X = np.hstack([blocks[k] for k in keys])
            for mtag, mfn in MODELS.items():
                oof_pred[(fs, mtag)] = oof(sub, X, y, mfn)
        # also provide DeepOC via each regressor (calibration reference)
        Xd = blocks['deepoc']
        for mtag, mfn in MODELS.items():
            oof_pred[('DeepOC', mtag)] = oof(sub, Xd, y, mfn)

        for (fs, reg), p in oof_pred.items():
            pt = r2(y, p)
            bvals = boot_r2(y, p, idx)
            lo, hi = ci(bvals)
            is_primary = PRIMARY.get(fs) == reg
            perf_rows.append(dict(subset=key, subset_label=label, n=n,
                                  feature_set=fs, regressor=reg,
                                  r2=pt, ci_lo=lo, ci_hi=hi, is_primary=int(is_primary)))
            # store draws only for primary models + all DeepOC to keep file lean
            if is_primary:
                for j, v in enumerate(bvals):
                    draw_rows.append(dict(subset=key, feature_set=fs, regressor=reg,
                                          draw=j, r2=v))

        # ---- paired deltas (same bootstrap indices) ----
        def prim(fs):
            reg = PRIMARY[fs]
            return oof_pred[(fs, reg)]

        comparisons = {
            'DeepOC_minus_features': (prim('DeepOC'), prim('features')),
            'combined_minus_DeepOC': (prim('DeepOC+features'), prim('DeepOC')),
            'ceiling_minus_DeepOC':  ('__ceiling__', prim('DeepOC')),
        }
        for cname, (a, b) in comparisons.items():
            if isinstance(a, str) and a == '__ceiling__':
                a_b = rho2_b                       # conservative ceiling
                a_pt = rho ** 2
            else:
                a_b, a_pt = boot_r2(y, a, idx), r2(y, a)
            b_b, b_pt = boot_r2(y, b, idx), r2(y, b)
            d_b = a_b - b_b
            delta_rows.append(dict(subset=key, comparison=cname,
                                   delta_r2=a_pt - b_pt,
                                   ci_lo=ci(d_b)[0], ci_hi=ci(d_b)[1],
                                   frac_positive=float((d_b > 0).mean())))

        # ---- per-row OOF predictions for scatter plots ----
        out = sub[['Spacer', 'Target', 'Fold', 'Info', 'Replicate 1', 'Replicate 2']].copy()
        out['activity'] = y
        out['pred_deepoc'] = oof_pred[('DeepOC', 'raw')]
        out['pred_features_gbm'] = oof_pred[('features', 'GBM')]
        out['pred_combined_gbm'] = oof_pred[('DeepOC+features', 'GBM')]
        out.to_csv(f'{OUTDIR}/oof_predictions_{key}.csv', index=False)

    pd.DataFrame(perf_rows).to_csv(f'{OUTDIR}/model_performance.csv', index=False)
    pd.DataFrame(ceil_rows).to_csv(f'{OUTDIR}/reproducibility_ceiling.csv', index=False)
    pd.DataFrame(delta_rows).to_csv(f'{OUTDIR}/paired_deltas.csv', index=False)
    pd.DataFrame(draw_rows).to_csv(f'{OUTDIR}/bootstrap_draws.csv', index=False)

    with open(f'{OUTDIR}/README.txt', 'w') as f:
        f.write(README)
    print('written to', OUTDIR, flush=True)


README = """\
Feature-vs-DeepOC vs reproducibility-ceiling analysis — output files
====================================================================

model_performance.csv
    One row per (subset, feature_set, regressor).
    r2                out-of-fold coefficient of determination on Fold0..4
    ci_lo, ci_hi      95% bootstrap CI (B=2000, percentile)
    is_primary        1 = the regressor used as the headline for that feature set
                      (GBM for feature models; 'raw' for DeepOC = its own OOF
                      prediction, no refit)
    feature_set values (handcrafted features = the ones characterized in the
    manuscript: GC content, positional nucleotide composition, and PAM):
        GC                raw GC content of spacer and target
        positional        position x nucleotide one-hot of the protospacer
        PAM               PAM 4-nt position x nucleotide one-hot (16 features)
        features          GC + positional + PAM (all handcrafted features)
        DeepOC            final sequence-only DeepOC model, out-of-fold prediction
        DeepOC+features   DeepOC prediction plus the handcrafted features
    regressor values: RF (random forest), GBM (hist. gradient boosting),
        lin (ridge), raw (no model; DeepOC OOF prediction used directly)

reproducibility_ceiling.csv
    Replicate-based upper bound on achievable R^2, per subset. The target
    (`activity`) is a single background-subtracted value rather than the mean of
    the two replicates, so the ceiling band is [rho^2, rho]:
    rho                        Pearson r between Replicate 1 and Replicate 2;
                               reliability of a single measurement
    r2_ceiling                 = rho (upper bound; max R^2 for a single-
                               measurement target under classical test theory)
    r2_ceiling_conservative    rho^2  (R^2 of one experiment predicting an
                               independent repeat; conservative lower bound)
    *_lo/_hi                   95% bootstrap CIs

paired_deltas.csv
    Paired bootstrap (same resample indices within a subset) for:
    DeepOC_minus_features   DeepOC (raw) minus features=GC+positional+PAM (GBM)
    combined_minus_DeepOC   DeepOC+features (GBM) minus DeepOC (raw)
    ceiling_minus_DeepOC    conservative ceiling (rho^2) minus DeepOC (raw)
    delta_r2, ci_lo, ci_hi  point estimate and 95% CI of the difference
    frac_positive           fraction of bootstrap resamples with delta > 0

bootstrap_draws.csv
    Long-format raw bootstrap R^2 draws for the primary model of each feature
    set (columns: subset, feature_set, regressor, draw, r2). Use for violin /
    custom interval plots.

oof_predictions_<subset>.csv
    Per-row out-of-fold predictions for scatter/agreement plots:
    activity (measured), pred_deepoc, pred_features_gbm (GC+positional+PAM),
    pred_combined_gbm, plus Replicate 1/2 and the Info label.

Subsets: matched_NGG (on-target, NGG PAM; primary), matched_nonNGG
    (on-target, non-NGG PAM), matched_all (on-target, all PAMs).
Rows are restricted to the Fold0..4 cross-validation partition.
"""


if __name__ == '__main__':
    main()
