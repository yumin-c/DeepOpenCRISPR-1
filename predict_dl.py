"""
DeepOC — inference with the final OpenCRISPR-1 activity model.

The model predicts day-7 editing activity (%) from sequence alone; the only
required input columns are Spacer (19-nt) and Target (30-nt). Predictions are the
mean of the five cross-validation fold models (averaged in log space).

Usage:
    python predict_dl.py --input data/demo_input.tsv --output predictions.csv
    python predict_dl.py --input my_guides.tsv --output out.csv --model_dir results/deepoc
"""
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_dl import DeepOC, DeepOCDataset, SEQ_LENGTH

N_FOLDS = 5
BATCH_SIZE = 256


def load_fold_model(model_dir, fold, device):
    model = DeepOC(dropout_rate=0.2)
    state = torch.load(f'{model_dir}/fold{fold}.pt', map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(data, model_dir, device):
    # DeepOCDataset expects an activity column; add a dummy one if absent.
    if 'OpenCRISPR-1 activity (day 7, %)' not in data.columns:
        data = data.assign(**{'OpenCRISPR-1 activity (day 7, %)': 0.0})
    dl = DataLoader(DeepOCDataset(data), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)
    logs = []
    for fold in range(N_FOLDS):
        model = load_fold_model(model_dir, fold, device)
        p = []
        with torch.no_grad():
            for sp, tg, *_ in dl:
                p.append(model(sp.to(device), tg.to(device)).squeeze().cpu().numpy())
        logs.append(np.concatenate(p))
    return np.expm1(np.mean(logs, axis=0))


def main():
    ap = argparse.ArgumentParser(description='DeepOC inference')
    ap.add_argument('--input', required=True, help='Input TSV with Spacer and Target columns')
    ap.add_argument('--output', default='predictions.csv')
    ap.add_argument('--model_dir', default='results/deepoc')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    data = pd.read_csv(args.input, sep='\t')
    missing = [c for c in ('Spacer', 'Target') if c not in data.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    print(f'Input samples: {len(data)}')

    data['prediction'] = predict(data, args.model_dir, device)
    print(f'Prediction range: [{data["prediction"].min():.2f}, {data["prediction"].max():.2f}]')
    data.to_csv(args.output, index=False)
    print(f'Saved to: {args.output}')


if __name__ == '__main__':
    main()
