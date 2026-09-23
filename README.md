# DeepOpenCRISPR-1 — OpenCRISPR-1 Activity Prediction

A deep-learning regressor for OpenCRISPR-1 guide-RNA activity (day 7, %),
predicted **from sequence alone**. DeepOpenCRISPR-1 pairs a convolutional trunk
over the spacer/target one-hot with a dedicated branch that reads the 4-nt PAM
at full resolution. The repository also contains the conventional-ML baselines
and a variance analysis quantifying how much of the per-target activity
variance the characterized sequence features explain relative to the model.

## 1. System requirements

### Software dependencies

Required Python packages (any reasonably recent version should work; the
versions below are the lower bounds we have validated):

| Package        | Minimum version | Used for |
|----------------|-----------------|----------|
| Python         | 3.10            | Runtime  |
| PyTorch        | 2.0             | DeepOpenCRISPR-1 training/inference |
| NumPy          | 1.23            | Tensor / array ops |
| pandas         | 1.5             | Data I/O |
| scikit-learn   | 1.2             | ML baselines, variance analysis |
| SciPy          | 1.10            | Spearman / Pearson |
| matplotlib     | 3.6             | Plots |
| seaborn        | 0.12            | Variance-analysis figure |
| XGBoost        | 1.7             | ML baseline |
| LightGBM       | 3.3             | ML baseline |
| CatBoost       | 1.2             | ML baseline |

### Tested on

- Ubuntu 22.04 / RHEL 9
- Python 3.11, PyTorch 2.x with CUDA 12.x

### Hardware

- A single NVIDIA GPU is sufficient for training (the model is small; < 4 GB
  VRAM). Inference runs on CPU as well.
- No non-standard hardware is required.

## 2. Installation guide

```bash
git clone https://github.com/yumin-c/DeepOpenCRISPR-1.git
cd DeepOpenCRISPR-1

# Create an environment (conda example)
conda create -n deepopencrispr python=3.11 -y
conda activate deepopencrispr

# Install PyTorch matching your CUDA version (see https://pytorch.org)
pip install torch

# Other dependencies (exact versions used for the deposited results)
pip install -r requirements.txt
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

A CSV identical to the input plus a final `prediction` column with the predicted
activity (%, original scale) from the 5-fold ensemble. Predictions for the 180
demo rows fall in roughly `[0, 90]` %, matching the training distribution.

### Expected runtime

- CPU (no GPU): **< 30 seconds** for 180 samples.
- GPU: **< 5 seconds** (most of the time is spent loading the 5 fold models).

## 4. Instructions for use

### Predict on your own data

DeepOpenCRISPR-1 uses **sequence only** — the input needs just two columns:

| Column | Description |
|--------|-------------|
| `Spacer` | 19-nt guide-RNA spacer |
| `Target` | 30-nt target (5-nt upstream + 19-nt protospacer + 3-nt PAM + 3-nt downstream) |

Ground-truth activity is **not** required at inference.

```bash
python predict_dl.py --input your_data.tsv --output your_predictions.csv
# optional: --model_dir results/deepoc   --device cpu
```

### Reproducing manuscript results

The training data is at
[data/OpenCRISPR-1_dataset.tsv](data/OpenCRISPR-1_dataset.tsv)
(13,943 sgRNA–target pairs; 5 CV folds + held-out test set).

```bash
# 1. Train the final DeepOpenCRISPR-1 (5-fold CV + held-out test) -> results/deepoc/
python train_dl.py

# 2. Conventional ML baselines (19 models, 5-fold CV, one-hot input)
python train_ml.py

# 3. Held-out test evaluation of the ML baselines
python predict_ml.py

# 4. Variance analysis: how much activity variance the sequence features
#    (GC, positional, PAM) explain versus DeepOpenCRISPR-1, with a
#    replicate-based ceiling
python variance_analysis/feature_ceiling_report.py    # blocks vs model vs ceiling
python variance_analysis/analyze_feature_combos.py    # 7 feature combinations
python variance_analysis/analyze_paper_features.py    # per-feature variance (sublibrary)
```

The trained fold models and out-of-fold predictions live in
[results/deepoc/](results/deepoc/) and are used by `predict_dl.py` and by the
variance analysis. Numeric variance-analysis outputs (with 95 % bootstrap
confidence intervals) are written into `variance_analysis/`; see
[variance_analysis/README.txt](variance_analysis/README.txt) for the column
dictionary.

### Model inputs

Both model families are trained on sequence alone, with no computed sequence
features (GC content, melting temperature or folding free energy):

| | Input |
|---|---|
| DeepOpenCRISPR-1 | spacer + target one-hot (4 × 30 each), plus the 4-nt PAM window |
| Conventional ML  | 196-dim flat one-hot (spacer 19 × 4 = 76, target 30 × 4 = 120) |

### Headline results

Prediction accuracy (Pearson r):

| Model | 5-fold CV | Held-out test |
|-------|-----------|---------------|
| **DeepOpenCRISPR-1** | **0.927** | **0.932** |
| Best conventional ML (XGBoost) | 0.680 | 0.723 |

The conventional baselines span 19 algorithms — linear and regularized models,
linear SVR, k-nearest neighbours, decision-tree, bagging and boosting
ensembles, and a multilayer perceptron. Per-model and per-fold numbers are in
[results/ml_260920_1223/](results/ml_260920_1223/): `cv_summary_onehot_only.csv`
(mean ± s.d. across folds), `cv_per_fold_onehot_only.csv` (individual folds),
and `test_summary_onehot_only.csv` (held-out test).

## File layout

```
OC1/
├── train_dl.py             DeepOpenCRISPR-1 training (5-fold CV + test ensemble)
├── predict_dl.py           DeepOpenCRISPR-1 inference (Spacer + Target only)
├── train_ml.py             19 conventional ML baselines (one-hot input)
├── predict_ml.py           ML-baseline inference on the held-out test set
├── plot_comparison.py      DL-vs-ML comparison plots
├── data/
│   ├── OpenCRISPR-1_dataset.tsv          main dataset (13,943 pairs; activity + folds)
│   ├── OpenCRISPR-1_replicates.csv       same pairs + replicate measurements (for the ceiling)
│   ├── OpenCRISPR-1_feature_library.csv  PAM / spacer-length characterization sublibrary
│   └── demo_input.tsv                    180-sample demo
├── results/
│   ├── deepoc/             fold weights (fold0..4.pt), CV / test predictions and metrics
│   └── ml_260920_1223/     ML baselines: CV summary, per-fold and per-model predictions
├── variance_analysis/      sequence-feature variance vs the model (+ reproducibility ceiling)
│   ├── feature_ceiling_report.py      feature blocks vs model vs ceiling (bootstrap)
│   ├── analyze_feature_combos.py      7 feature-block combinations, per stratum
│   ├── analyze_paper_features.py      per-feature variance on the characterization sublibrary
│   ├── analyze_ceiling_vs_deepoc.py   shared helpers (loading, out-of-fold, bootstrap)
│   ├── *.csv                          numeric results with 95 % bootstrap CIs
│   └── README.txt                     column dictionary
├── requirements.txt        pinned dependency versions
└── LICENSE
```

## License

Released under the [MIT License](LICENSE).
