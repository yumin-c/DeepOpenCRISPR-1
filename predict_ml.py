"""
ML Test Set Prediction

Trains each ML model on the full train/val data, predicts on test set.
Saves per-model test predictions and summary metrics.
"""
import os
import numpy as np
import pandas as pd
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

FEATURE_COLS = [
    'GC_spacer', 'GC_target', 'GC_target_5p_context', 'GC_target_PAM_distal',
    'GC_target_PAM_proximal', 'GC_target_PAM', 'GC_target_3p_context',
    'Tm_spacer', 'Tm_target', 'Tm_target_5p_context', 'Tm_target_PAM_distal',
    'Tm_target_PAM_proximal', 'Tm_target_PAM', 'Tm_target_3p_context',
    'MFE_spacer', 'MFE_sgRNA',
]

SAVE_DIR = 'results/ml_260211_1744'


def encode_sequences_onehot(spacers, targets):
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
        'CatBoost': CatBoostRegressor(iterations=300, depth=6, learning_rate=0.1, random_seed=42, verbose=0),
        'MLP': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
    }


def main():
    data = pd.read_csv('data/20260129_DeepOpenCRISPR-1_sequence_with_features.tsv', sep='\t')
    train_val = data[data['Fold'] != 'Test'].copy()
    test = data[data['Fold'] == 'Test'].copy()
    print(f'Train/Val: {len(train_val)}, Test: {len(test)}')

    y_train = train_val['OpenCRISPR-1 activity (day 7, %)'].values
    y_test = test['OpenCRISPR-1 activity (day 7, %)'].values

    for mode in ['features_only', 'onehot_features']:
        print(f'\n{"="*60}')
        print(f'  MODE: {mode}')
        print(f'{"="*60}')

        if mode == 'features_only':
            X_train = train_val[FEATURE_COLS].values
            X_test = test[FEATURE_COLS].values
        else:
            oh_train = encode_sequences_onehot(train_val['Spacer'].values, train_val['Target'].values)
            oh_test = encode_sequences_onehot(test['Spacer'].values, test['Target'].values)
            X_train = np.hstack([oh_train, train_val[FEATURE_COLS].values])
            X_test = np.hstack([oh_test, test[FEATURE_COLS].values])

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        models_dict = get_models()
        rows = []

        for name, model in models_dict.items():
            try:
                model.fit(X_train_sc, y_train)
                pred = model.predict(X_test_sc)
                sr, _ = spearmanr(y_test, pred)
                pr = pearsonr(y_test, pred).correlation
                rows.append({'model': name, 'spearman': sr, 'pearson': pr})
                print(f'  {name:25s} | S={sr:.4f} | P={pr:.4f}')

                # Save per-model predictions
                pred_df = test[['Spacer', 'Target', 'OpenCRISPR-1 activity (day 7, %)']].copy()
                pred_df['prediction'] = pred
                pred_df.to_csv(os.path.join(SAVE_DIR, f'test_pred_{mode}_{name}.csv'), index=False)
            except Exception as e:
                print(f'  {name:25s} | ERROR: {e}')
                rows.append({'model': name, 'spearman': np.nan, 'pearson': np.nan})

        summary = pd.DataFrame(rows).sort_values('spearman', ascending=False)
        summary.to_csv(os.path.join(SAVE_DIR, f'test_summary_{mode}.csv'), index=False)

    print(f'\nResults saved to: {SAVE_DIR}/')


if __name__ == '__main__':
    main()
