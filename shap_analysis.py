import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import spearmanr, pearsonr
from datetime import datetime

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

SHAP_MODELS = {
    'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                            random_state=42, tree_method='hist', n_jobs=-1, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                               random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=300, depth=6, learning_rate=0.1,
                                   random_seed=42, verbose=0),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=20,
                                          random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                   learning_rate=0.1, random_state=42),
}


def main():
    data = pd.read_csv('data/20260129_DeepOpenCRISPR-1_sequence_with_features.tsv', sep='\t')
    train_val = data[data['Fold'] != 'Test'].copy()
    test = data[data['Fold'] == 'Test'].copy()

    X_train = train_val[FEATURE_COLS].values
    y_train = train_val['OpenCRISPR-1 activity (day 7, %)'].values
    X_test = test[FEATURE_COLS].values
    y_test = test['OpenCRISPR-1 activity (day 7, %)'].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    timestamp = datetime.now().strftime('%y%m%d_%H%M')
    save_dir = f'results/shap_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    for model_name, model in SHAP_MODELS.items():
        print(f'\n===== {model_name} =====')

        # Train on all train data
        model.fit(X_train_sc, y_train)
        pred_test = model.predict(X_test_sc)
        sr, _ = spearmanr(y_test, pred_test)
        pr = pearsonr(y_test, pred_test).correlation
        print(f'  Test: Spearman={sr:.4f}, Pearson={pr:.4f}')

        # Retrain with unscaled features for interpretable SHAP
        if model_name == 'CatBoost':
            model_unscaled = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.1,
                                                random_seed=42, verbose=0)
        else:
            model_unscaled = SHAP_MODELS[model_name].__class__(
                **{k: v for k, v in model.get_params().items() if v is not None}
            )
        model_unscaled.fit(X_train, y_train)
        X_display = pd.DataFrame(X_test, columns=FEATURE_COLS)

        if model_name == 'XGBoost':
            # XGBoost 3.x + SHAP compatibility: use predict-based explainer
            background = shap.maskers.Independent(X_train[:200], max_samples=200)
            explainer = shap.Explainer(model_unscaled.predict, background)
            shap_values_obj = explainer(X_display)
            shap_values = shap_values_obj.values
        else:
            explainer = shap.TreeExplainer(model_unscaled)
            shap_values = explainer.shap_values(X_test)

        # 1) Summary (bee swarm) plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_display, show=False, max_display=16)
        plt.title(f'{model_name} — SHAP Summary')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_summary_{model_name}.jpg'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2) Bar plot (mean |SHAP|)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_display, plot_type='bar', show=False, max_display=16)
        plt.title(f'{model_name} — Mean |SHAP| Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_bar_{model_name}.jpg'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3) Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
        shap_df.to_csv(os.path.join(save_dir, f'shap_values_{model_name}.csv'), index=False)

        # 4) Feature importance ranking
        importance = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame({
            'feature': FEATURE_COLS,
            'mean_abs_shap': importance,
        }).sort_values('mean_abs_shap', ascending=False)
        imp_df.to_csv(os.path.join(save_dir, f'shap_importance_{model_name}.csv'), index=False)
        print(f'  Top 5 features:')
        for _, row in imp_df.head(5).iterrows():
            print(f'    {row["feature"]:30s} | {row["mean_abs_shap"]:.4f}')

    # ── Combined importance comparison across models ──
    combined_rows = []
    for model_name in SHAP_MODELS:
        imp_file = os.path.join(save_dir, f'shap_importance_{model_name}.csv')
        if os.path.exists(imp_file):
            imp = pd.read_csv(imp_file)
            imp['model'] = model_name
            combined_rows.append(imp)

    if combined_rows:
        combined = pd.concat(combined_rows)
        pivot = combined.pivot(index='feature', columns='model', values='mean_abs_shap')
        pivot['mean_across_models'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('mean_across_models', ascending=False)
        pivot.to_csv(os.path.join(save_dir, 'shap_importance_combined.csv'))

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_data = pivot.drop(columns='mean_across_models')
        im = ax.imshow(plot_data.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(plot_data.columns)))
        ax.set_xticklabels(plot_data.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(plot_data.index)))
        ax.set_yticklabels(plot_data.index, fontsize=9)
        plt.colorbar(im, label='Mean |SHAP|')
        ax.set_title('Feature Importance Comparison Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_importance_heatmap.jpg'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f'\nAll SHAP results saved to: {save_dir}/')


if __name__ == '__main__':
    main()
