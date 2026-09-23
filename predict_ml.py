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
    data = pd.read_csv('data/OpenCRISPR-1_dataset.tsv', sep='\t')
    train_val = data[data['Fold'] != 'Test'].copy()
    test = data[data['Fold'] == 'Test'].copy()
    print(f'Train/Val: {len(train_val)}, Test: {len(test)}')

    y_train = train_val['OpenCRISPR-1 activity (day 7, %)'].values
    y_test = test['OpenCRISPR-1 activity (day 7, %)'].values

    print(f'\n{"="*60}')
    print('  19 conventional ML models | one-hot input (196 dims)')
    print(f'{"="*60}')

    X_train = encode_sequences_onehot(train_val['Spacer'].values, train_val['Target'].values)
    X_test = encode_sequences_onehot(test['Spacer'].values, test['Target'].values)

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
            pred_df.to_csv(os.path.join(SAVE_DIR, f'test_pred_onehot_only_{name}.csv'), index=False)
        except Exception as e:
            print(f'  {name:25s} | ERROR: {e}')
            rows.append({'model': name, 'spearman': np.nan, 'pearson': np.nan})

    summary = pd.DataFrame(rows).sort_values('spearman', ascending=False)
    summary.to_csv(os.path.join(SAVE_DIR, 'test_summary_onehot_only.csv'), index=False)

    print(f'\nResults saved to: {SAVE_DIR}/')


if __name__ == '__main__':
    main()
