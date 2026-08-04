"""
Variance explained by the sequence features characterized in the manuscript.

Reviewer point 6 asks how much of the target-to-target activity variance is
accounted for by the features described in Fig. 2d-h: spacer GC content, spacer
length, position-specific nucleotide composition, and PAM. This script encodes
those features as described in the text and reports cross-validated R^2 for the
full set and the incremental contribution of each.

Two encodings of the positional signal are compared:
  motif      - only the specific biases stated in the text (G at positions
               8-20, T depletion, C at positions 2/3, purine at position 20)
  positional - the full position x nucleotide profile

The gap between them is the part of the positional signal the manuscript
describes only qualitatively.

Position numbering follows the manuscript: position 1 is PAM-proximal, so
manuscript position p maps to target index 24 - p. Position 20 is the target
base pairing with the guide's additional 5' guanine.

Note on libraries: the genomic library carries only NGGN PAMs (the optimal
class), so PAM identity is near-constant there; the NNNN PAM library is the
only place PAM class can be estimated. They are analysed separately.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import os
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data', 'OpenCRISPR-1_feature_library.csv')
ACT = 'OpenCRISPR-1 activity (day 7, %)'
NTS = 'ACGT'
SEED = 0
N_SPLITS = 5

PROTO_POS = list(range(1, 21))          # manuscript positions 1..20
tgt_idx = lambda p: 24 - p              # manuscript position -> target index


def pos_onehot(tgt, positions):
    X = np.zeros((len(tgt), len(positions) * 4), dtype=np.float32)
    for i, s in enumerate(tgt):
        for j, p in enumerate(positions):
            k = NTS.find(s[tgt_idx(p)])
            if k >= 0:
                X[i, j * 4 + k] = 1.0
    return X


def motif_features(tgt):
    """Only the positional biases the manuscript states explicitly."""
    cols = {}
    seqs = [[s[tgt_idx(p)] for p in PROTO_POS] for s in tgt]   # index 0 -> pos 1
    arr = np.array(seqs)

    g_span = [p for p in range(8, 21)]                      # G enriched, pos 8-20
    cols['G_count_pos8_20'] = np.isin(arr[:, [p - 1 for p in g_span]], ['G']).sum(1)
    cols['T_count_all'] = (arr == 'T').sum(1)               # T broadly depleted
    distal = [p for p in range(11, 21)]                     # PAM-distal half
    cols['A_count_distal'] = (arr[:, [p - 1 for p in distal]] == 'A').sum(1)
    cols['C_pos2'] = (arr[:, 1] == 'C').astype(float)       # C enriched at 2, 3
    cols['C_pos3'] = (arr[:, 2] == 'C').astype(float)
    cols['C_count_pos5_6_8'] = (arr[:, [4, 5, 7]] == 'C').sum(1)
    cols['purine_pos20'] = np.isin(arr[:, 19], ['A', 'G']).astype(float)
    return pd.DataFrame(cols).values.astype(np.float32)


def gc_features(df):
    """Raw GC content of spacer and target."""
    return np.column_stack([
        df['Spacer [GC]'].values.astype(np.float32),
        df['Target [GC]'].values.astype(np.float32),
    ])


def pam_features(df):
    """PAM as N1/N2/N3/N4 position x nucleotide one-hot (16 features)."""
    pam = df['4-nt PAM'].values
    X = [np.array([[1.0 if s[i] == nt else 0.0 for nt in NTS] for s in pam],
                  dtype=np.float32) for i in range(4)]
    return np.hstack(X)


def cv_r2(X, y, model_fn):
    if X.shape[1] == 0 or np.allclose(X.std(0), 0):
        return 0.0
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)


RIDGE = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
GBM = lambda: HistGradientBoostingRegressor(random_state=SEED)


def report(label, blocks, y):
    print(f'\n===== {label} (n={len(y)}) =====')
    for tag, fn in [('linear (ridge)', RIDGE), ('nonlinear (GBM)', GBM)]:
        full = np.hstack(list(blocks.values()))
        r2_full = cv_r2(full, y, fn)
        print(f'\n  -- {tag} --   full model R^2 = {r2_full:.4f}')
        print(f"     {'block':<22}{'alone':>9}{'incremental':>14}")
        for b in blocks:
            alone = cv_r2(blocks[b], y, fn)
            rest = [v for k, v in blocks.items() if k != b]
            inc = r2_full - (cv_r2(np.hstack(rest), y, fn) if rest else 0.0)
            print(f'     {b:<22}{alone:>9.4f}{inc:>+14.4f}')


def main():
    df = pd.read_csv(DATA)
    n_per = df.groupby('Spacer')['Target'].transform('size')
    gen, pam_lib = df[n_per == 1].copy(), df[n_per > 1].copy()

    # --- genomic library, 19-nt guides: the per-target variance question ---
    g = gen[gen['Spacer length'] == 19]
    y = g[ACT].values.astype(float)
    report('genomic library, 19-nt guides -- manuscript motifs only', {
        'GC': gc_features(g),
        'motif': motif_features(g.Target.values),
        'PAM': pam_features(g),
    }, y)
    report('genomic library, 19-nt guides -- full positional profile', {
        'GC': gc_features(g),
        'positional': pos_onehot(g.Target.values, PROTO_POS),
        'PAM': pam_features(g),
    }, y)

    # --- spacer length: paired design, same target, 15-23 nt ---
    print('\n===== spacer length (genomic library, all lengths) =====')
    fam = gen[gen.Spacer.str.len().groupby(gen.Target).transform('size') > 0]
    by_len = gen.groupby('Spacer length')[ACT].agg(['mean', 'std', 'size'])
    print(by_len.round(2).to_string())

    # --- PAM library: the only place PAM class varies ---
    yp = pam_lib[ACT].values.astype(float)
    report('PAM library (NNNN, fixed spacers)', {
        'GC': gc_features(pam_lib),
        'positional': pos_onehot(pam_lib.Target.values, PROTO_POS),
        'PAM': pam_features(pam_lib),
    }, yp)


if __name__ == '__main__':
    main()
