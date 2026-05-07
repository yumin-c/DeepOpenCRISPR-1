import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from datetime import datetime

FEATURE_COLS = [
    'GC_spacer', 'GC_target', 'GC_target_5p_context', 'GC_target_PAM_distal',
    'GC_target_PAM_proximal', 'GC_target_PAM', 'GC_target_3p_context',
    'Tm_spacer', 'Tm_target', 'Tm_target_5p_context', 'Tm_target_PAM_distal',
    'Tm_target_PAM_proximal', 'Tm_target_PAM', 'Tm_target_3p_context',
    'MFE_spacer', 'MFE_sgRNA',
]
SEQ_LENGTH = 30
SEED = 216


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
    def __init__(self, data, feature_mean=None, feature_std=None):
        self.indices = list(data.index)
        self.spacers = np.stack([one_hot_encode(s, SEQ_LENGTH, pad='center') for s in data['Spacer']])
        self.targets = np.stack([one_hot_encode(s, SEQ_LENGTH, pad='left') for s in data['Target']])

        features = data[FEATURE_COLS].values.astype(np.float32)
        if feature_mean is None:
            self.feature_mean = features.mean(axis=0)
            self.feature_std = features.std(axis=0) + 1e-8
        else:
            self.feature_mean = feature_mean
            self.feature_std = feature_std
        self.features = (features - self.feature_mean) / self.feature_std

        self.activity = np.log1p(data['OpenCRISPR-1 activity (day 7, %)'].values).astype(np.float32)
        self.ontarget = (data['Spacer'].values == data['Target'].str[5:24].values).astype(np.int64)

    def __len__(self):
        return len(self.activity)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.spacers[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.activity[idx]),
            torch.tensor(self.ontarget[idx]),
        )


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


class BalancedMSELoss(nn.Module):
    def __init__(self, on_weight=0.5, off_weight=1.0):
        super().__init__()
        self.on_weight = on_weight
        self.off_weight = off_weight
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, actual, ontarget):
        pred = pred.view(-1)
        y = actual.view(-1)
        on_mask = ontarget == 1
        off_mask = ~on_mask
        loss = 0.0
        if on_mask.sum() > 0:
            loss += self.mse(pred[on_mask], y[on_mask]) * self.on_weight
        if off_mask.sum() > 0:
            loss += self.mse(pred[off_mask], y[off_mask]) * self.off_weight
        return loss / pred.size(0)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_fold(train_data, val_data, config, device):
    set_seed(SEED)

    train_ds = DeepOCDataset(train_data)
    val_ds = DeepOCDataset(val_data,
                           feature_mean=train_ds.feature_mean,
                           feature_std=train_ds.feature_std)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    model = DeepOC(n_features=len(FEATURE_COLS), dropout_rate=config['dropout_rate']).to(device)
    criterion = BalancedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config['T_0'], T_mult=2, eta_min=config['learning_rate'] / 100
    )

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for spacer, target, feats, activity, ontarget in train_loader:
            spacer, target, feats = spacer.to(device), target.to(device), feats.to(device)
            activity, ontarget = activity.to(device), ontarget.to(device)

            optimizer.zero_grad()
            pred = model(spacer, target, feats)
            loss = criterion(pred.squeeze(), activity, ontarget)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        preds, gts, ons = [], [], []
        with torch.no_grad():
            for spacer, target, feats, activity, ontarget in val_loader:
                spacer, target, feats = spacer.to(device), target.to(device), feats.to(device)
                activity = activity.to(device)
                pred = model(spacer, target, feats)
                loss = criterion(pred.squeeze(), activity, ontarget.to(device))
                val_loss += loss.item()
                preds.append(pred.squeeze().cpu())
                gts.append(activity.cpu())
                ons.append(ontarget)

        pred_ = torch.cat(preds).numpy()
        gt_ = torch.cat(gts).numpy()
        on_ = torch.cat(ons).numpy()

        sr, _ = spearmanr(pred_, gt_)
        pr = pearsonr(pred_, gt_).correlation

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  E {epoch+1:3d} | S {sr:.4f} | P {pr:.4f} | VL {val_loss/len(val_loader):.4f} | TL {train_loss/len(train_loader):.4f}')

    pred_orig = np.expm1(pred_)
    gt_orig = np.expm1(gt_)
    sr_orig, _ = spearmanr(pred_orig, gt_orig)
    pr_orig = pearsonr(pred_orig, gt_orig).correlation

    on_mask = on_ == 1
    off_mask = ~on_mask
    metrics = {'spearman': sr_orig, 'pearson': pr_orig}
    if on_mask.sum() > 0:
        metrics['on_spearman'], _ = spearmanr(pred_orig[on_mask], gt_orig[on_mask])
        metrics['on_pearson'] = pearsonr(pred_orig[on_mask], gt_orig[on_mask]).correlation
    if off_mask.sum() > 0:
        metrics['off_spearman'], _ = spearmanr(pred_orig[off_mask], gt_orig[off_mask])
        metrics['off_pearson'] = pearsonr(pred_orig[off_mask], gt_orig[off_mask]).correlation

    return model, train_ds.feature_mean, train_ds.feature_std, val_ds.indices, pred_orig, metrics


def predict_test(models, test_data, train_datasets_stats, device, config):
    all_preds = []
    for model, (feat_mean, feat_std) in zip(models, train_datasets_stats):
        model.eval()
        test_ds = DeepOCDataset(test_data, feature_mean=feat_mean, feature_std=feat_std)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        preds = []
        with torch.no_grad():
            for spacer, target, feats, *_ in test_loader:
                spacer, target, feats = spacer.to(device), target.to(device), feats.to(device)
                pred = model(spacer, target, feats).squeeze().cpu().numpy()
                preds.append(pred)
        all_preds.append(np.concatenate(preds))

    ensemble_log = np.mean(all_preds, axis=0)
    ensemble_orig = np.expm1(ensemble_log)
    return ensemble_orig


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    config = {
        'learning_rate': 2e-3,
        'dropout_rate': 0.2,
        'batch_size': 64,
        'epochs': 70,
        'T_0': 10,
    }

    data = pd.read_csv('data/20260129_DeepOpenCRISPR-1_sequence_with_features.tsv', sep='\t')
    train_val_data = data[data['Fold'] != 'Test'].copy()
    test_data = data[data['Fold'] == 'Test'].copy()
    print(f"Train/Val: {len(train_val_data)}, Test: {len(test_data)}")

    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    save_dir = f'results/dl_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    models = []
    train_stats = []
    all_cv_metrics = []
    cv_predictions = train_val_data.copy()
    cv_predictions['prediction'] = np.nan

    for fold in range(5):
        print(f'\n===== FOLD {fold} =====')
        train_data = train_val_data[train_val_data['Fold'] != f'Fold{fold}']
        val_data = train_val_data[train_val_data['Fold'] == f'Fold{fold}']

        model, feat_mean, feat_std, val_indices, val_preds, metrics = train_fold(train_data, val_data, config, device)

        models.append(model)
        train_stats.append((feat_mean, feat_std))

        cv_predictions.loc[val_indices, 'prediction'] = val_preds
        all_cv_metrics.append(metrics)

        print(f'  Fold {fold} | Spearman={metrics["spearman"]:.4f} | Pearson={metrics["pearson"]:.4f}')
        if 'on_spearman' in metrics:
            print(f'         | On  S={metrics["on_spearman"]:.4f} P={metrics["on_pearson"]:.4f}')
            print(f'         | Off S={metrics["off_spearman"]:.4f} P={metrics["off_pearson"]:.4f}')

        torch.save({
            'state_dict': model.state_dict(),
            'feature_mean': feat_mean,
            'feature_std': feat_std,
        }, os.path.join(save_dir, f'fold{fold}.pt'))

    metrics_df = pd.DataFrame(all_cv_metrics)
    metrics_df.index.name = 'fold'
    metrics_df.to_csv(os.path.join(save_dir, 'cv_metrics.csv'))
    print('\n===== 5-Fold CV Summary =====')
    for col in ['spearman', 'pearson', 'on_spearman', 'on_pearson', 'off_spearman', 'off_pearson']:
        if col in metrics_df.columns:
            print(f'  {col:15s}: {metrics_df[col].mean():.4f} +/- {metrics_df[col].std():.4f}')

    cv_predictions.to_csv(os.path.join(save_dir, 'cv_predictions.csv'), index=False)

    print('\n===== Test Set Prediction =====')
    test_preds = predict_test(models, test_data, train_stats, device, config)
    test_data = test_data.copy()
    test_data['prediction'] = test_preds
    gt = test_data['OpenCRISPR-1 activity (day 7, %)'].values

    sr_test, _ = spearmanr(gt, test_preds)
    pr_test = pearsonr(gt, test_preds).correlation
    print(f'  Total  | Spearman={sr_test:.4f} | Pearson={pr_test:.4f}')

    ontarget_mask = test_data['Spacer'].values == test_data['Target'].str[5:24].values
    if ontarget_mask.sum() > 0:
        sr_on, _ = spearmanr(gt[ontarget_mask], test_preds[ontarget_mask])
        pr_on = pearsonr(gt[ontarget_mask], test_preds[ontarget_mask]).correlation
        print(f'  On-tgt | Spearman={sr_on:.4f} | Pearson={pr_on:.4f} (n={ontarget_mask.sum()})')
    if (~ontarget_mask).sum() > 0:
        sr_off, _ = spearmanr(gt[~ontarget_mask], test_preds[~ontarget_mask])
        pr_off = pearsonr(gt[~ontarget_mask], test_preds[~ontarget_mask]).correlation
        print(f'  Off-tgt| Spearman={sr_off:.4f} | Pearson={pr_off:.4f} (n={(~ontarget_mask).sum()})')

    test_data.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)

    test_metrics = {'spearman': sr_test, 'pearson': pr_test}
    if ontarget_mask.sum() > 0:
        test_metrics.update({'on_spearman': sr_on, 'on_pearson': pr_on})
    if (~ontarget_mask).sum() > 0:
        test_metrics.update({'off_spearman': sr_off, 'off_pearson': pr_off})
    pd.DataFrame([test_metrics]).to_csv(os.path.join(save_dir, 'test_metrics.csv'), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.ravel()
    for fold in range(5):
        fold_data = cv_predictions[cv_predictions['Fold'] == f'Fold{fold}']
        gt_f = fold_data['OpenCRISPR-1 activity (day 7, %)']
        pr_f = fold_data['prediction']
        sr_f, _ = spearmanr(gt_f, pr_f)
        pr_f_corr = pearsonr(gt_f, pr_f).correlation
        ax = axes_flat[fold]
        ax.scatter(gt_f, pr_f, s=0.5, alpha=0.5)
        ax.plot([0, 90], [0, 90], 'k--', alpha=0.3)
        ax.set_xlim([0, 90])
        ax.set_ylim([0, 90])
        ax.set_title(f'Fold{fold} (S={sr_f:.4f}, r={pr_f_corr:.4f})')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
    axes_flat[5].remove()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_fold_scatter.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mask, title in [
        (axes[0], np.ones(len(test_data), dtype=bool), 'All'),
        (axes[1], ontarget_mask, 'On-Target'),
        (axes[2], ~ontarget_mask, 'Off-Target'),
    ]:
        if mask.sum() == 0:
            continue
        g = gt[mask]
        p = test_preds[mask]
        sr_p, _ = spearmanr(g, p)
        pr_p = pearsonr(g, p).correlation
        ax.scatter(g, p, s=1, alpha=0.5)
        ax.plot([0, 90], [0, 90], 'k--', alpha=0.3)
        ax.set_xlim([0, 90])
        ax.set_ylim([0, 90])
        ax.set_title(f'{title} (n={mask.sum()})\nS={sr_p:.4f}, r={pr_p:.4f}')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_scatter.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = [metrics_df[c].values for c in ['on_spearman', 'off_spearman', 'on_pearson', 'off_pearson'] if c in metrics_df.columns]
    box_labels = [c for c in ['on_spearman', 'off_spearman', 'on_pearson', 'off_pearson'] if c in metrics_df.columns]
    colors = ['#e74c3c', '#e67e22', '#3498db', '#1abc9c']
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    for i, d in enumerate(box_data):
        ax.scatter([i + 1] * len(d), d, color='black', alpha=0.7, edgecolors='white', zorder=3)
    ax.set_xticklabels(box_labels, rotation=15)
    ax.set_ylabel('Correlation')
    ax.set_title('DeepOC — 5-Fold CV On/Off Target Metrics')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_boxplot.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\nAll results saved to: {save_dir}/')


if __name__ == '__main__':
    main()
