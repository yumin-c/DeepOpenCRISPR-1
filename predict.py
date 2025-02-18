import os
import torch
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import CRISPRDataset, CRISPRNet
from datetime import datetime
from scipy.stats import spearmanr, pearsonr

def load_models(model_dir, device):
    models = []
    for fold in range(5):
        model_path = os.path.join(model_dir, f"fold{fold}.pt")
        model = torch.load(model_path, map_location=device)
        model.eval()
        models.append(model)
    return models

def predict(models, test_loader, device):
    predictions = []
    with torch.no_grad():
        for spacer, target, *_ in test_loader:
            spacer, target = spacer.to(device), target.to(device)
            fold_preds = [model(spacer, target).squeeze().cpu().numpy() for model in models]
            predictions.extend(np.mean(fold_preds, axis=0))
    predictions = np.exp(predictions)-1
    return np.array(predictions)

import matplotlib.pyplot as plt

def plot_on_off_target(data, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    
    # On-target data
    on_target = data[data['ontarget'] == 1]
    off_target = data[data['ontarget'] == 0]
    
    # Spearman and Pearson correlations
    sr_on, _ = spearmanr(on_target['OpenCRISPR-1 activity (day 7, %)'], on_target['prediction'])
    pr_on = pearsonr(on_target['OpenCRISPR-1 activity (day 7, %)'], on_target['prediction']).correlation
    sr_off, _ = spearmanr(off_target['OpenCRISPR-1 activity (day 7, %)'], off_target['prediction'])
    pr_off = pearsonr(off_target['OpenCRISPR-1 activity (day 7, %)'], off_target['prediction']).correlation
    
    # Plot On-target
    axes[0].scatter(on_target['OpenCRISPR-1 activity (day 7, %)'], on_target['prediction'], s=1)
    axes[0].plot([0, 70], [0, 70], 'k--', alpha=0.3)
    axes[0].set_xlim([0, 70])
    axes[0].set_ylim([0, 70])
    axes[0].set_title(f'On-Target (n={len(on_target)})\nR = {sr_on:.4f}, r = {pr_on:.4f}')
    axes[0].set_xlabel('Ground Truth')
    axes[0].set_ylabel('Prediction')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot Off-target
    axes[1].scatter(off_target['OpenCRISPR-1 activity (day 7, %)'], off_target['prediction'], s=1)
    axes[1].plot([0, 70], [0, 70], 'k--', alpha=0.3)
    axes[1].set_xlim([0, 70])
    axes[1].set_ylim([0, 70])
    axes[1].set_title(f'Off-Target (n={len(off_target)})\nR = {sr_off:.4f}, r = {pr_off:.4f}')
    axes[1].set_xlabel('Ground Truth')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plot_save_path = save_path.replace('.csv', '_ontarget_offtarget_plot.jpg')
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"On/Off target plots saved to: {plot_save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save predictions")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data = pd.read_csv(args.file_path)
    data = data[data['Fold']=='Test']
    data["Activity"] = data["OpenCRISPR-1 activity (day 7, %)"]
    test_dataset = CRISPRDataset(data)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    models = load_models('/extdata2/YMC/OC1/saved_models/250212_1420', device)
    data["prediction"] = predict(models, test_loader, device)

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(args.save_dir, f"prediction_{timestamp}.csv")
    data.to_csv(save_path, index=False, 
                columns=["Spacer", "Target", "OpenCRISPR-1 activity (day 7, %)", 
                        "Fold", "ontarget", "prediction"])
    print(f"Predictions saved to: {save_path}")

    plot_on_off_target(data, save_path)

if __name__ == "__main__":
    main()