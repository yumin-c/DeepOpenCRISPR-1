import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor,
)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

FEATURE_COLS = [
    'GC_spacer', 'GC_target', 'GC_target_5p_context', 'GC_target_PAM_distal',
    'GC_target_PAM_proximal', 'GC_target_PAM', 'GC_target_3p_context',
    'Tm_spacer', 'Tm_target', 'Tm_target_5p_context', 'Tm_target_PAM_distal',
    'Tm_target_PAM_proximal', 'Tm_target_PAM', 'Tm_target_3p_context',
    'MFE_spacer', 'MFE_sgRNA',
]


def encode_sequences_onehot(spacers, targets):
    """Spacer (19x4=76) + Target (30x4=120) = 196-dim flat one-hot."""
    bases = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    rows = []
    for sp, tg in zip(spacers, targets):
        vec = []
        for seq in [sp, tg]:
            for base in seq:
                oh = [0, 0, 0, 0]
                if base in bases:
                    oh[bases[base]] = 1
                vec.extend(oh)
        rows.append(vec)
    return np.array(rows, dtype=np.float32)


def get_models():
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        'BayesianRidge': BayesianRidge(),
        'SGD': SGDRegressor(max_iter=5000, random_state=42),
        'LinearSVR': LinearSVR(C=1.0, max_iter=10000, random_state=42),
        'KNN_5': KNeighborsRegressor(n_neighbors=5),
        'KNN_10': KNeighborsRegressor(n_neighbors=10),
        'DecisionTree': DecisionTreeRegressor(max_depth=15, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'Bagging': BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42,
                                tree_method='hist', n_jobs=-1, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42,
                                   n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=300, depth=6, learning_rate=0.1, random_seed=42,
                                       verbose=0),
        'MLP': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
    }


def calc_metrics(y_true, y_pred):
    sr, _ = spearmanr(y_true, y_pred)
    pr = pearsonr(y_true, y_pred).correlation
    return sr, pr


def run_cv(data, feature_mode='features_only'):
    """feature_mode: 'features_only' (16 dims) or 'onehot_features' (196+16=212 dims)."""
    folds = [f'Fold{i}' for i in range(5)]
    models_dict = get_models()

    results = {name: [] for name in models_dict}
    all_predictions = {name: {} for name in models_dict}

    for fold_idx, fold_name in enumerate(folds):
        print(f'\n--- {fold_name} ---')
        train = data[data['Fold'] != fold_name]
        val = data[data['Fold'] == fold_name]

        if feature_mode == 'features_only':
            X_train = train[FEATURE_COLS].values
            X_val = val[FEATURE_COLS].values
        else:
            oh_train = encode_sequences_onehot(train['Spacer'].values, train['Target'].values)
            oh_val = encode_sequences_onehot(val['Spacer'].values, val['Target'].values)
            feat_train = train[FEATURE_COLS].values
            feat_val = val[FEATURE_COLS].values
            X_train = np.hstack([oh_train, feat_train])
            X_val = np.hstack([oh_val, feat_val])

        y_train = train['OpenCRISPR-1 activity (day 7, %)'].values
        y_val = val['OpenCRISPR-1 activity (day 7, %)'].values

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        for name, model in models_dict.items():
            try:
                model.fit(X_train_sc, y_train)
                pred = model.predict(X_val_sc)
                sr, pr = calc_metrics(y_val, pred)
                results[name].append({'fold': fold_idx, 'spearman': sr, 'pearson': pr})
                all_predictions[name][fold_name] = pred
                print(f'  {name:25s} | S={sr:.4f} | P={pr:.4f}')
            except Exception as e:
                print(f'  {name:25s} | ERROR: {e}')
                results[name].append({'fold': fold_idx, 'spearman': np.nan, 'pearson': np.nan})

        models_dict = get_models()

    return results, all_predictions


def main():
    data = pd.read_csv('data/20260129_DeepOpenCRISPR-1_sequence_with_features.tsv', sep='\t')
    train_val_data = data[data['Fold'] != 'Test'].copy()
    print(f'Train/Val samples: {len(train_val_data)}')

    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    save_dir = f'results/ml_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    print('\n' + '='*60)
    print('  MODE: 16 Computed Features Only')
    print('='*60)
    results_feat, preds_feat = run_cv(train_val_data, feature_mode='features_only')

    print('\n' + '='*60)
    print('  MODE: One-Hot + 16 Features')
    print('='*60)
    results_oh, preds_oh = run_cv(train_val_data, feature_mode='onehot_features')

    def summarize(results, mode_name):
        rows = []
        for name, fold_results in results.items():
            df = pd.DataFrame(fold_results)
            rows.append({
                'model': name,
                'mean_spearman': df['spearman'].mean(),
                'std_spearman': df['spearman'].std(),
                'mean_pearson': df['pearson'].mean(),
                'std_pearson': df['pearson'].std(),
            })
        summary = pd.DataFrame(rows).sort_values('mean_spearman', ascending=False)
        summary.to_csv(os.path.join(save_dir, f'cv_summary_{mode_name}.csv'), index=False)
        return summary

    for mode_name, results in [('features_only', results_feat), ('onehot_features', results_oh)]:
        all_rows = []
        for name, fold_results in results.items():
            for r in fold_results:
                all_rows.append({'model': name, **r})
        pd.DataFrame(all_rows).to_csv(os.path.join(save_dir, f'cv_per_fold_{mode_name}.csv'), index=False)

    summary_feat = summarize(results_feat, 'features_only')
    summary_oh = summarize(results_oh, 'onehot_features')

    print('\n===== CV Summary (Features Only) =====')
    print(summary_feat.to_string(index=False))
    print('\n===== CV Summary (One-Hot + Features) =====')
    print(summary_oh.to_string(index=False))

    for mode_name, preds in [('features_only', preds_feat), ('onehot_features', preds_oh)]:
        for model_name, fold_preds in preds.items():
            pred_df = train_val_data.copy()
            pred_df['prediction'] = np.nan
            for fold_name, p in fold_preds.items():
                mask = pred_df['Fold'] == fold_name
                pred_df.loc[mask, 'prediction'] = p
            pred_df.to_csv(
                os.path.join(save_dir, f'cv_pred_{mode_name}_{model_name}.csv'),
                index=False,
                columns=['Spacer', 'Target', 'OpenCRISPR-1 activity (day 7, %)', 'Fold', 'prediction']
            )

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, summary, title in [
        (axes[0], summary_feat, '16 Features Only'),
        (axes[1], summary_oh, 'One-Hot + 16 Features'),
    ]:
        x = np.arange(len(summary))
        width = 0.35
        ax.barh(x - width/2, summary['mean_spearman'], width, xerr=summary['std_spearman'],
                label='Spearman', color='#3498db', alpha=0.8, capsize=3)
        ax.barh(x + width/2, summary['mean_pearson'], width, xerr=summary['std_pearson'],
                label='Pearson', color='#e74c3c', alpha=0.8, capsize=3)
        ax.set_yticks(x)
        ax.set_yticklabels(summary['model'], fontsize=9)
        ax.set_xlabel('Correlation')
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_comparison_barplot.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\nAll results saved to: {save_dir}/')


if __name__ == '__main__':
    main()
