"""
SHAP GradientExplainer analysis for DeepOC (deep learning model).
Analyzes feature importance including sequence features (per-position and per-base).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from datetime import datetime

# ──────────────────────────────────────────────
# Constants (same as train_dl.py)
# ──────────────────────────────────────────────
FEATURE_COLS = [
    'GC_spacer', 'GC_target', 'GC_target_5p_context', 'GC_target_PAM_distal',
    'GC_target_PAM_proximal', 'GC_target_PAM', 'GC_target_3p_context',
    'Tm_spacer', 'Tm_target', 'Tm_target_5p_context', 'Tm_target_PAM_distal',
    'Tm_target_PAM_proximal', 'Tm_target_PAM', 'Tm_target_3p_context',
    'MFE_spacer', 'MFE_sgRNA',
]
SEQ_LENGTH = 30
BASES = ['A', 'T', 'G', 'C']

# ──────────────────────────────────────────────
# Model (same as train_dl.py)
# ──────────────────────────────────────────────
class DeepOC(nn.Module):
    def __init__(self, n_features=16, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feature_branch = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.GELU(),
        )
        self.dense = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 7 + 32, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, spacer, target, features):
        x = torch.cat((spacer, target), dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        f = self.feature_branch(features)
        x = torch.cat((x, f), dim=1)
        return self.dense(x)


# ──────────────────────────────────────────────
# Preprocessing (same as train_dl.py)
# ──────────────────────────────────────────────
def one_hot_encode(seq, length, pad='center'):
    bases = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    encoding = np.zeros((4, length), dtype=np.float32)
    seq_len = len(seq)
    if pad == 'left':
        start = length - seq_len
    elif pad == 'center':
        start = (length - seq_len) // 2
    else:
        start = 0
    for i, base in enumerate(seq):
        if base in bases:
            encoding[bases[base], start + i] = 1
    return encoding


def prepare_inputs(data, feature_mean, feature_std):
    spacers = np.stack([one_hot_encode(s, SEQ_LENGTH, pad='center') for s in data['Spacer']])
    targets = np.stack([one_hot_encode(s, SEQ_LENGTH, pad='left') for s in data['Target']])
    features = data[FEATURE_COLS].values.astype(np.float32)
    features = (features - feature_mean) / feature_std
    return (
        torch.tensor(spacers),
        torch.tensor(targets),
        torch.tensor(features),
    )


# ──────────────────────────────────────────────
# Model wrapper for SHAP (single-call interface)
# ──────────────────────────────────────────────
class DeepOCWrapper(nn.Module):
    """Wraps DeepOC to accept a single concatenated tensor for SHAP.

    Input layout (along dim=1):
      [0:4*SEQ_LENGTH]              -> spacer one-hot  (4 x 30 = 120, flattened)
      [4*SEQ_LENGTH:8*SEQ_LENGTH]   -> target one-hot  (4 x 30 = 120, flattened)
      [8*SEQ_LENGTH:]               -> 16 features
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        B = x.size(0)
        spacer = x[:, :4 * SEQ_LENGTH].view(B, 4, SEQ_LENGTH)
        target = x[:, 4 * SEQ_LENGTH:8 * SEQ_LENGTH].view(B, 4, SEQ_LENGTH)
        features = x[:, 8 * SEQ_LENGTH:]
        return self.model(spacer, target, features)


def pack_inputs(spacer_t, target_t, feature_t):
    """Flatten and concatenate model inputs into a single 2D tensor."""
    B = spacer_t.size(0)
    return torch.cat([
        spacer_t.view(B, -1),   # (B, 120)
        target_t.view(B, -1),   # (B, 120)
        feature_t,               # (B, 16)
    ], dim=1)


# ──────────────────────────────────────────────
# SHAP Analysis
# ──────────────────────────────────────────────
def compute_shap_values(wrapped_model, background_packed, explain_packed, device, batch_size=128):
    """Compute SHAP values using GradientExplainer with batched explanation."""
    explainer = shap.GradientExplainer(wrapped_model, background_packed)

    n = explain_packed.size(0)
    all_shap = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = explain_packed[start:end]
        sv = explainer.shap_values(batch)
        # sv: list of arrays or single array -> shape (B, input_dim)
        if isinstance(sv, list):
            sv = sv[0]
        all_shap.append(sv)
        print(f'  SHAP progress: {end}/{n}')
    shap_out = np.concatenate(all_shap, axis=0)  # (N, 256) or (N, 256, 1)
    if shap_out.ndim == 3:
        shap_out = shap_out.squeeze(-1)
    return shap_out  # (N, 256)


def unpack_shap(shap_vals):
    """Split SHAP values back into spacer, target, feature components."""
    spacer_shap = shap_vals[:, :4 * SEQ_LENGTH].reshape(-1, 4, SEQ_LENGTH)   # (N,4,30)
    target_shap = shap_vals[:, 4 * SEQ_LENGTH:8 * SEQ_LENGTH].reshape(-1, 4, SEQ_LENGTH)  # (N,4,30)
    feature_shap = shap_vals[:, 8 * SEQ_LENGTH:]                              # (N,16)
    return spacer_shap, target_shap, feature_shap


# ──────────────────────────────────────────────
# Unified signed-SHAP table (CSV)
# ──────────────────────────────────────────────
def build_unified_signed_shap(spacer_shap, target_shap, feature_shap):
    """
    Build a single comparable table of mean signed SHAP for every input dimension.

    Metric: mean signed SHAP across all test samples (logo-style).
      - Sequence (spacer / target): shap.mean(axis=0) → (4, 30)
        Label format: "Sp_A_P6", "Tg_G_P22"  (target coordinate 1-30)
      - Bio-features: feature_shap.mean(axis=0) → (16,)
        Label: feature name as-is

    Returns a DataFrame with columns:
        label, mean_signed_shap, abs_shap, type
    sorted by abs_shap descending.
    """
    rows = []

    # Sequence: mean signed SHAP per (base, position)
    for seq_tag, shap_arr in [('Sp', spacer_shap), ('Tg', target_shap)]:
        logo = shap_arr.mean(axis=0)  # (4, 30)
        for b_idx, base in enumerate(BASES):
            for p in range(SEQ_LENGTH):
                rows.append({
                    'label': f'{seq_tag}_{base}_P{p + 1}',
                    'mean_signed_shap': float(logo[b_idx, p]),
                    'type': 'spacer' if seq_tag == 'Sp' else 'target',
                })

    # Bio-features: mean signed SHAP
    feat_logo = feature_shap.mean(axis=0)  # (16,)
    for i, name in enumerate(FEATURE_COLS):
        rows.append({
            'label': name,
            'mean_signed_shap': float(feat_logo[i]),
            'type': 'biofeature',
        })

    df = pd.DataFrame(rows)
    df['abs_shap'] = df['mean_signed_shap'].abs()
    return df.sort_values('abs_shap', ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load data ──
    data = pd.read_csv('data/20260129_DeepOpenCRISPR-1_sequence_with_features.tsv', sep='\t')
    test_data = data[data['Fold'] == 'Test'].copy().reset_index(drop=True)
    train_val_data = data[data['Fold'] != 'Test'].copy().reset_index(drop=True)
    print(f'Test: {len(test_data)}, Train/Val: {len(train_val_data)}')

    # ── Load fold models ──
    model_dir = 'results/dl_260211_1719'
    fold_models = []
    fold_stats = []  # per-fold (feature_mean, feature_std)
    for fold in range(5):
        ckpt = torch.load(os.path.join(model_dir, f'fold{fold}.pt'), map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model = DeepOC(n_features=len(FEATURE_COLS)).to(device)
            model.load_state_dict(ckpt['state_dict'])
            fold_stats.append((
                ckpt.get('feature_mean', None),
                ckpt.get('feature_std', None),
            ))
        else:
            # full model object
            model = ckpt
            fold_stats.append((None, None))
        model.eval()
        fold_models.append(model)
    print(f'Loaded {len(fold_models)} fold models')

    # ── Output directory ──
    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    save_dir = f'results/shap_unified_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    print(f'Output: {save_dir}')

    # ── Prepare background indices (same for all folds) ──
    np.random.seed(42)
    bg_idx = np.random.choice(len(train_val_data), size=200, replace=False)
    bg_data = train_val_data.iloc[bg_idx].reset_index(drop=True)

    # ── Run SHAP per fold using fold-specific normalization stats ──
    all_shap_vals = []
    for fold, (model, (feat_mean, feat_std)) in enumerate(zip(fold_models, fold_stats)):
        print(f'\n--- Fold {fold} SHAP ---')
        # Use fold's own normalization stats
        if feat_mean is None:
            features_all = train_val_data[FEATURE_COLS].values.astype(np.float32)
            feat_mean = features_all.mean(axis=0)
            feat_std = features_all.std(axis=0) + 1e-8

        bg_spacer, bg_target, bg_feat = prepare_inputs(bg_data, feat_mean, feat_std)
        bg_packed = pack_inputs(bg_spacer, bg_target, bg_feat).to(device)

        te_spacer, te_target, te_feat = prepare_inputs(test_data, feat_mean, feat_std)
        te_packed = pack_inputs(te_spacer, te_target, te_feat).to(device)

        if fold == 0:
            print(f'  Background: {bg_packed.shape}, Explain: {te_packed.shape}')

        wrapped = DeepOCWrapper(model).to(device)
        wrapped.eval()
        shap_vals = compute_shap_values(wrapped, bg_packed, te_packed, device, batch_size=256)
        all_shap_vals.append(shap_vals)
        np.save(os.path.join(save_dir, f'shap_values_fold{fold}.npy'), shap_vals)

    # Ensemble: average SHAP across folds
    ensemble_shap = np.mean(all_shap_vals, axis=0)  # (N, 256)
    np.save(os.path.join(save_dir, 'shap_values_ensemble.npy'), ensemble_shap)
    print(f'\nSHAP values shape: {ensemble_shap.shape}')

    # ── Unpack ──
    spacer_shap, target_shap, feature_shap = unpack_shap(ensemble_shap)
    print(f'spacer_shap: {spacer_shap.shape}, target_shap: {target_shap.shape}, feature_shap: {feature_shap.shape}')

    # ── Save unified CSV ──
    df_unified = build_unified_signed_shap(spacer_shap, target_shap, feature_shap)
    csv_path = os.path.join(save_dir, 'shap_unified_signed.csv')
    df_unified.to_csv(csv_path, index=False)
    print(f'\nSaved: {csv_path}')

    # ── Print top entries ──
    print('\n===== Top 20 features by |mean signed SHAP| =====')
    for rank, row in df_unified.head(20).iterrows():
        sign = '+' if row['mean_signed_shap'] >= 0 else '-'
        print(f'  {rank+1:2d}. {row["label"]:30s}  {sign}{abs(row["mean_signed_shap"]):.4f}  [{row["type"]}]')

    print(f'\nAll results saved to: {save_dir}/')
    print(f'Run:  python plot_shap_dl.py {save_dir}')


if __name__ == '__main__':
    main()
