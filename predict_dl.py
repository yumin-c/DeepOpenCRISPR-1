"""
DeepOC — Inference Script

Usage:
    python predict_dl.py --input data/input.tsv --output results/predictions.csv

Input TSV columns (same format as training data, without Fold):
    Spacer, Target, GC_spacer, GC_target, GC_target_5p_context, GC_target_PAM_distal,
    GC_target_PAM_proximal, GC_target_PAM, GC_target_3p_context,
    Tm_spacer, Tm_target, Tm_target_5p_context, Tm_target_PAM_distal,
    Tm_target_PAM_proximal, Tm_target_PAM, Tm_target_3p_context,
    MFE_spacer, MFE_sgRNA
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
FEATURE_COLS = [
    'GC_spacer', 'GC_target', 'GC_target_5p_context', 'GC_target_PAM_distal',
    'GC_target_PAM_proximal', 'GC_target_PAM', 'GC_target_3p_context',
    'Tm_spacer', 'Tm_target', 'Tm_target_5p_context', 'Tm_target_PAM_distal',
    'Tm_target_PAM_proximal', 'Tm_target_PAM', 'Tm_target_3p_context',
    'MFE_spacer', 'MFE_sgRNA',
]
SEQ_LENGTH = 30
MODEL_DIR = 'results/dl_260211_1719'
N_FOLDS = 5
BATCH_SIZE = 256


# ──────────────────────────────────────────────
# Model
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
        x = torch.cat((spacer, target), dim=1)   # (B, 8, 30)
        x = self.conv(x)                          # (B, 128, 7)
        x = x.view(x.size(0), -1)                # (B, 896)
        f = self.feature_branch(features)          # (B, 32)
        x = torch.cat((x, f), dim=1)              # (B, 928)
        return self.dense(x)


# ──────────────────────────────────────────────
# Dataset
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


class DeepOCDataset(Dataset):
    def __init__(self, data, feature_mean, feature_std):
        self.spacers = np.stack([one_hot_encode(s, SEQ_LENGTH, pad='center') for s in data['Spacer']])
        self.targets = np.stack([one_hot_encode(s, SEQ_LENGTH, pad='left') for s in data['Target']])

        features = data[FEATURE_COLS].values.astype(np.float32)
        self.features = (features - feature_mean) / feature_std

    def __len__(self):
        return len(self.spacers)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.spacers[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.features[idx]),
        )


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def load_fold_model(fold, device):
    ckpt = torch.load(f'{MODEL_DIR}/fold{fold}.pt', map_location=device, weights_only=False)
    model = DeepOC(n_features=len(FEATURE_COLS), dropout_rate=0.2)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt['feature_mean'], ckpt['feature_std']


def predict(data, device='cuda'):
    fold_preds = []

    for fold in range(N_FOLDS):
        model, feat_mean, feat_std = load_fold_model(fold, device)
        ds = DeepOCDataset(data, feat_mean, feat_std)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        preds = []
        with torch.no_grad():
            for spacer, target, feats in loader:
                spacer, target, feats = spacer.to(device), target.to(device), feats.to(device)
                pred = model(spacer, target, feats).squeeze().cpu().numpy()
                preds.append(pred)
        fold_preds.append(np.concatenate(preds))

    # Ensemble: average in log-space, then expm1
    ensemble_log = np.mean(fold_preds, axis=0)
    return np.expm1(ensemble_log)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='DeepOC Inference')
    parser.add_argument('--input', required=True, help='Input TSV file path')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV file path')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu, auto-detect if not set)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load input data
    data = pd.read_csv(args.input, sep='\t')
    print(f'Input samples: {len(data)}')

    # Validate required columns
    required = ['Spacer', 'Target'] + FEATURE_COLS
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    # Predict
    predictions = predict(data, device)
    data['prediction'] = predictions
    print(f'Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]')

    # Save
    data.to_csv(args.output, index=False)
    print(f'Saved to: {args.output}')


if __name__ == '__main__':
    main()
