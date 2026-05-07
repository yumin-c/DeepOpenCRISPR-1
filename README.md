# DeepOC (DeepOpenCRISPR) — OpenCRISPR-1 Activity Prediction

A deep-learning regressor for OpenCRISPR-1 guide-RNA activity (day 7, %),
benchmarked against 19 conventional ML baselines and explained with SHAP.

## 1. System requirements

### Software dependencies

Required Python packages (any reasonably recent version should work; the
versions below are the lower bounds we have validated):

| Package        | Minimum version | Used for |
|----------------|-----------------|----------|
| Python         | 3.10            | Runtime  |
| PyTorch        | 2.0             | DeepOC training/inference |
| NumPy          | 1.23            | Tensor / array ops |
| pandas         | 1.5             | Data I/O |
| scikit-learn   | 1.2             | ML baselines, scaling |
| SciPy          | 1.10            | Spearman / Pearson |
| matplotlib     | 3.6             | Plots |
| XGBoost        | 1.7             | ML baseline |
| LightGBM       | 3.3             | ML baseline |
| CatBoost       | 1.2             | ML baseline |
| SHAP           | 0.42            | Interpretation |
| Biopython      | 1.80            | Feature engineering (only needed if recomputing features) |
| ViennaRNA      | 2.6             | MFE feature (only needed if recomputing features) |

### Tested on

- Ubuntu 22.04 / RHEL 9
- Python 3.11, PyTorch 2.x with CUDA 12.x

### Hardware

- A single NVIDIA GPU with >= 24 GB VRAM is sufficient for training (e.g.
  RTX 3090 / 4090 / A5000). Inference runs on CPU as well.
- No non-standard hardware is required.

## 2. Installation guide

```bash
git clone https://github.com/yumin-c/DeepOpenCRISPR.git
cd DeepOpenCRISPR

# Create an environment (conda example)
conda create -n deepoc python=3.11 -y
conda activate deepoc

# Install PyTorch matching your CUDA version (see https://pytorch.org)
pip install torch

# Other dependencies
pip install numpy pandas scikit-learn scipy matplotlib \
            xgboost lightgbm catboost shap biopython
```

Typical install time on a normal desktop: **5–10 minutes**, dominated by the
PyTorch download.

## 3. Demo

A small, real-data demo is provided in [data/demo_input.tsv](data/demo_input.tsv)
(180 rows, 30 each from Fold0..Fold4 and Test).

### Run

```bash
python predict_dl.py --input data/demo_input.tsv --output demo_predictions.csv
```

### Expected output

A CSV identical to the input plus a final `prediction` column containing the
predicted activity (%, original scale) from the 5-fold ensemble. Example
header:

```
Spacer  Target  OpenCRISPR-1 activity (day 7, %)  Fold  ...  prediction
```

Predictions for the 180 demo rows fall in roughly `[0, 90]` %, matching the
training distribution.

### Expected runtime

- CPU (no GPU): **< 30 seconds** for 180 samples.
- GPU: **< 5 seconds** (most of the time is spent loading the 5 fold models).

## 4. Instructions for use

### Predict on your own data

Prepare a tab-separated file with the columns below, then run
`predict_dl.py`:

| Column | Description |
|--------|-------------|
| `Spacer` | 19-bp guide-RNA spacer |
| `Target` | 30-bp target sequence (5-bp upstream + 19-bp protospacer + 3-bp PAM + 3-bp downstream) |
| 7 `GC_*` columns | GC content of spacer / target windows |
| 7 `Tm_*` columns | Melting temperature of spacer / target windows |
| `MFE_spacer`, `MFE_sgRNA` | RNA minimum free energy (ViennaRNA) |

The 16 numerical features are listed at the top of `train_dl.py`
(`FEATURE_COLS`). Activity ground truth is **not** required at inference.

```bash
python predict_dl.py --input your_data.tsv --output your_predictions.csv
# optional: --device cpu
```

### Reproducing manuscript results

The full training data is at
[data/20260129_DeepOpenCRISPR-1_sequence_with_features.tsv](data/20260129_DeepOpenCRISPR-1_sequence_with_features.tsv)
(13,943 samples; 5 CV folds + held-out test set).

```bash
# 1. DeepOC: 5-fold CV + held-out test (~30 min on a single GPU)
python train_dl.py

# 2. 19 ML baselines: 5-fold CV in two feature modes (~30–60 min on CPU)
python train_ml.py

# 3. SHAP for tree-based ML models
python shap_analysis.py

# 4. SHAP for the DeepOC ensemble (writes results/shap_unified_*/)
python shap_dl_analysis.py
python plot_shap_dl.py            # plots from latest results/shap_unified_*

# 5. DL vs ML comparison plots
python plot_comparison.py
```

Outputs land in timestamped folders under `results/`. Pretrained DeepOC fold
models live in [results/dl_260211_1719/](results/dl_260211_1719/) and are
used by `predict_dl.py` and `shap_dl_analysis.py` by default.

### Headline results

5-fold CV (mean +/- std):

| Model | Spearman | Pearson |
|-------|----------|---------|
| **DeepOC** | **0.8835 +/- 0.017** | **0.9247 +/- 0.013** |
| CatBoost | 0.640 +/- 0.117 | 0.620 +/- 0.131 |
| XGBoost  | 0.634 +/- 0.107 | 0.655 +/- 0.096 |

Held-out test set:

| Model | Spearman | Pearson |
|-------|----------|---------|
| **DeepOC** | **0.9044** | **0.9363** |
| XGBoost | 0.689 | 0.709 |

## File layout

```
OC1/
├── train_dl.py             DeepOC training (5-fold CV + test ensemble)
├── train_ml.py             19 ML baselines (5-fold CV, two feature modes)
├── shap_analysis.py        SHAP for the 5 tree-based ML baselines
├── shap_dl_analysis.py     DeepGradientShap on the DeepOC fold ensemble
├── plot_shap_dl.py         Plotting helper for shap_dl_analysis output
├── plot_comparison.py      DL-vs-ML box-plot comparisons
├── predict_dl.py           DeepOC inference CLI
├── predict_ml.py           ML-baseline inference on the held-out test set
├── data/
│   ├── 20260129_DeepOpenCRISPR-1_sequence_with_features.tsv   full dataset
│   └── demo_input.tsv      180-sample demo subset
└── results/
    ├── dl_260211_1719/                pretrained DeepOC folds + metrics
    ├── ml_260211_1744/                ML baselines CV + test results
    ├── shap_unified_260403_1655/      DeepOC SHAP (manuscript figure)
    └── model_comparison_*.jpg         DL-vs-ML comparison plots
```

## License

Released under the [MIT License](LICENSE).
