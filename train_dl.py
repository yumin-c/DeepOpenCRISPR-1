"""
DeepOC — training the final OpenCRISPR-1 activity model.

DeepOC predicts day-7 editing activity (%) from sequence alone. A convolutional
trunk reads the concatenated spacer/target one-hot, and a dedicated branch reads
the 4-nt PAM at full resolution (so PAM identity is not diluted by pooling)
before a shared regression head.

Usage:
    python train_dl.py                       # trains 5 folds, writes results/deepoc/
    python train_dl.py --outdir results/deepoc

Outputs (in --outdir):
    fold{0..4}.pt            per-fold model weights
    cv_predictions.csv       out-of-fold predictions on the training folds
    test_predictions.csv     held-out test predictions (mean of the 5 folds)
    cv_metrics.csv           per-stratum cross-validation metrics
    test_metrics.csv         per-stratum test metrics
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr

SEQ_LENGTH = 30
SEED = 216
DATA = 'data/OpenCRISPR-1_dataset.tsv'
CONFIG = dict(learning_rate=2e-3, dropout_rate=0.2, batch_size=64, epochs=70, T_0=10)


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
def one_hot_encode(seq, length, pad='center'):
    bases = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    enc = np.zeros((4, length), dtype=np.float32)
    start = length - len(seq) if pad == 'left' else (length - len(seq)) // 2 if pad == 'center' else 0
    for i, b in enumerate(seq):
        if b in bases:
            enc[bases[b], start + i] = 1
    return enc


class DeepOCDataset(Dataset):
    """Sequence one-hot only: spacer (centre-padded) + target (left-padded)."""
    def __init__(self, data):
        self.indices = list(data.index)
        self.spacers = np.stack([one_hot_encode(s, SEQ_LENGTH, 'center') for s in data['Spacer']])
        self.targets = np.stack([one_hot_encode(s, SEQ_LENGTH, 'left') for s in data['Target']])
        self.activity = np.log1p(data['OpenCRISPR-1 activity (day 7, %)'].values).astype(np.float32)
        self.ontarget = (data['Spacer'].values == data['Target'].str[5:24].values).astype(np.int64)

    def __len__(self):
        return len(self.activity)

    def __getitem__(self, i):
        return (torch.from_numpy(self.spacers[i]), torch.from_numpy(self.targets[i]),
                torch.tensor(self.activity[i]), torch.tensor(self.ontarget[i]))


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class DeepOC(nn.Module):
    """Conv trunk over the spacer+target one-hot, plus a dedicated branch that
    reads the 4-nt PAM at full resolution; their features are concatenated before
    the regression head."""
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1), nn.GELU(), nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.GELU(), nn.AvgPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.GELU())
        self.pam = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(128 * 7 + 32, 64), nn.GELU(),
            nn.Dropout(dropout_rate), nn.Linear(64, 1), nn.Softplus())

    def forward(self, spacer, target):
        c = self.conv(torch.cat((spacer, target), dim=1))
        c = c.view(c.size(0), -1)
        pam = target[:, :, 24:28].reshape(target.size(0), -1)   # 4-nt PAM one-hot (16)
        return self.head(torch.cat([c, self.pam(pam)], dim=1))


class BalancedMSELoss(nn.Module):
    """MSE with separate weights for on-target and off-target (mismatched) pairs."""
    def __init__(self, on_weight=0.5, off_weight=1.0):
        super().__init__()
        self.on_weight, self.off_weight = on_weight, off_weight
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, actual, ontarget):
        pred, y = pred.view(-1), actual.view(-1)
        on = ontarget == 1
        loss = 0.0
        if on.sum() > 0:
            loss += self.mse(pred[on], y[on]) * self.on_weight
        if (~on).sum() > 0:
            loss += self.mse(pred[~on], y[~on]) * self.off_weight
        return loss / pred.size(0)


# ──────────────────────────────────────────────
# Training / evaluation
# ──────────────────────────────────────────────
def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)


def train_fold(train_data, val_data, cfg, device):
    set_seed(SEED)
    tl = DataLoader(DeepOCDataset(train_data), batch_size=cfg['batch_size'], shuffle=True,
                    num_workers=4, pin_memory=True)
    vd = DeepOCDataset(val_data)
    vl = DataLoader(vd, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    model = DeepOC(dropout_rate=cfg['dropout_rate']).to(device)
    crit = BalancedMSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cfg['T_0'], T_mult=2, eta_min=cfg['learning_rate'] / 100)
    for _ in range(cfg['epochs']):
        model.train()
        for sp, tg, act, on in tl:
            sp, tg, act, on = sp.to(device), tg.to(device), act.to(device), on.to(device)
            opt.zero_grad()
            crit(model(sp, tg).squeeze(), act, on).backward()
            opt.step()
        sch.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for sp, tg, *_ in vl:
            preds.append(model(sp.to(device), tg.to(device)).squeeze().cpu().numpy())
    return model, vd.indices, np.expm1(np.concatenate(preds))


def predict(models, data, device):
    dl = DataLoader(DeepOCDataset(data), batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    logs = []
    for model in models:
        model.eval()
        p = []
        with torch.no_grad():
            for sp, tg, *_ in dl:
                p.append(model(sp.to(device), tg.to(device)).squeeze().cpu().numpy())
        logs.append(np.concatenate(p))
    return np.expm1(np.mean(logs, axis=0))


def stratum(df):
    on = df['Spacer'].values == df['Target'].str[5:24].values
    ngg = df['Target'].str[25:27].values == 'GG'
    return np.where(~on, 'off-target', np.where(ngg, 'matched_NGG', 'matched_nonNGG'))


def metrics_by_stratum(df):
    y = df['OpenCRISPR-1 activity (day 7, %)'].values
    p = df['prediction'].values
    s = stratum(df)
    rows = []
    for name in ['matched_NGG', 'matched_nonNGG', 'off-target', 'ALL']:
        m = np.ones(len(df), bool) if name == 'ALL' else (s == name)
        if m.sum() < 5:
            continue
        yy, pp = y[m], p[m]
        r2 = 1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2)
        rows.append(dict(stratum=name, n=int(m.sum()), pearson=pearsonr(yy, pp)[0],
                         spearman=spearmanr(yy, pp)[0], r2=r2))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results/deepoc')
    ap.add_argument('--data', default=DATA)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.outdir, exist_ok=True)
    print(f'DeepOC training | device={device} | out={args.outdir}', flush=True)

    data = pd.read_csv(args.data, sep='\t')
    tv = data[data['Fold'] != 'Test'].copy()
    test = data[data['Fold'] == 'Test'].copy()

    models = []
    cvp = tv.copy(); cvp['prediction'] = np.nan
    for fold in range(5):
        tr = tv[tv['Fold'] != f'Fold{fold}']
        va = tv[tv['Fold'] == f'Fold{fold}']
        model, idx, vpred = train_fold(tr, va, CONFIG, device)
        models.append(model)
        cvp.loc[idx, 'prediction'] = vpred
        torch.save(model.state_dict(), os.path.join(args.outdir, f'fold{fold}.pt'))
        fm = cvp.loc[idx]
        pr = pearsonr(fm['OpenCRISPR-1 activity (day 7, %)'], fm['prediction'])[0]
        print(f'  fold {fold} | val Pearson={pr:.4f}', flush=True)

    cvp.to_csv(os.path.join(args.outdir, 'cv_predictions.csv'), index=False)
    test['prediction'] = predict(models, test, device)
    test.to_csv(os.path.join(args.outdir, 'test_predictions.csv'), index=False)

    cv_m = metrics_by_stratum(cvp); cv_m.to_csv(os.path.join(args.outdir, 'cv_metrics.csv'), index=False)
    te_m = metrics_by_stratum(test); te_m.to_csv(os.path.join(args.outdir, 'test_metrics.csv'), index=False)
    print('\n=== 5-fold CV (out-of-fold) ===', flush=True)
    print(cv_m.round(4).to_string(index=False), flush=True)
    print('\n=== Test ===', flush=True)
    print(te_m.round(4).to_string(index=False), flush=True)
    print(f'\nsaved to {args.outdir}/', flush=True)


if __name__ == '__main__':
    main()
