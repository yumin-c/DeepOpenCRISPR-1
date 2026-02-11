# DeepOC — OpenCRISPR-1 Activity Prediction

Predicts OpenCRISPR-1 guide RNA activity (day 7, %) using deep learning and conventional ML models.

## Dataset

- **Source**: `data/20260129_DeepOpenCRISPR-1_sequence_with_features.tsv`
- **Samples**: 13,943 (Train/Val: 12,475 across 5 folds, Test: 1,468)
- **On-target**: 11,727 (Spacer == Target[5:24]), **Off-target**: 2,216
- **Input**: Spacer (19bp) + Target (30bp) + 16 pre-computed features
- **Features**: GC content (7), melting temperature (7), minimum free energy (2)
  - Computed with BioPython 1.86, ViennaRNA 2.7.2

## Results (5-Fold CV)

| Model | Spearman | Pearson |
|-------|----------|---------|
| **DeepOC (DL)** | **0.8835 +/- 0.017** | **0.9247 +/- 0.013** |
| CatBoost | 0.640 +/- 0.117 | 0.620 +/- 0.131 |
| XGBoost | 0.634 +/- 0.107 | 0.655 +/- 0.096 |
| GradientBoosting | 0.615 +/- 0.100 | 0.635 +/- 0.099 |
| LightGBM | 0.613 +/- 0.122 | 0.639 +/- 0.109 |

**Test set (DeepOC)**: Spearman = 0.9044, Pearson = 0.9363

## Model Architecture (DeepOC)

```
Spacer (19bp, center-padded to 30) ─┐
                                     ├─ Concat ─ Conv1d(8,32) ─ GELU ─ AvgPool
Target (30bp, left-padded to 30)  ──┘            Conv1d(32,64) ─ GELU ─ AvgPool
                                                 Conv1d(64,128) ─ GELU ─ Flatten(896)
                                                                            │
16 Features ─ Linear(16,32) ─ GELU ─────────────────────────── Concat(928) ─┘
                                                                     │
                                             Dropout ─ Linear(928,64) ─ GELU
                                             Dropout ─ Linear(64,1) ─ Softplus
```

- **Loss**: BalancedMSELoss (on-target weight=0.5, off-target weight=1.0)
- **Target transform**: log1p for training, expm1 for output
- **Optimizer**: Adam (lr=2e-3)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Ensemble**: Average of 5 fold models in log-space

## SHAP Feature Importance (Top 5)

| Feature | Mean \|SHAP\| across 5 tree models |
|---------|------|
| GC_target_PAM | 3.76 |
| MFE_sgRNA | 3.42 |
| Tm_target_PAM | 2.71 |
| Tm_target_PAM_proximal | 2.43 |
| MFE_spacer | 1.71 |

## Inference

```bash
conda activate ym_pytorch
python predict_dl.py --input data/input.tsv --output results/predictions.csv
```

**Input TSV columns** (tab-separated):
- `Spacer`: 19bp guide RNA spacer sequence
- `Target`: 30bp target sequence
- 16 feature columns: `GC_spacer`, `GC_target`, `GC_target_5p_context`, `GC_target_PAM_distal`, `GC_target_PAM_proximal`, `GC_target_PAM`, `GC_target_3p_context`, `Tm_spacer`, `Tm_target`, `Tm_target_5p_context`, `Tm_target_PAM_distal`, `Tm_target_PAM_proximal`, `Tm_target_PAM`, `Tm_target_3p_context`, `MFE_spacer`, `MFE_sgRNA`

**Output**: Input CSV with appended `prediction` column (predicted activity %).

## File Structure

```
OC1/
├── train_dl.py             # DeepOC training (5-fold CV + test)
├── train_ml.py             # 19 ML models training (5-fold CV)
├── shap_analysis.py        # SHAP feature importance (5 tree models)
├── predict_dl.py           # DeepOC inference
├── plot_comparison.py      # DL vs ML box plot comparison
├── data/
│   └── 20260129_DeepOpenCRISPR-1_sequence_with_features.tsv
└── results/
    ├── dl_260211_1719/     # DL results, fold models, predictions
    ├── ml_260211_1744/     # ML CV results, predictions
    └── shap_260211_1756/   # SHAP values, summary/bar plots, heatmap
```
