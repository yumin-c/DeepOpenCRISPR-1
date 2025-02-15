import os
import torch
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader
from utils import CRISPRDataset, CRISPRNet
from datetime import datetime

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
        for spacer, target, _ in test_loader:
            spacer, target = spacer.to(device), target.to(device)
            fold_preds = [model(spacer, target).squeeze().cpu().numpy() for model in models]
            predictions.extend(np.mean(fold_preds, axis=0))
    
    predictions = np.exp(predictions)-1
    return np.array(predictions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save predictions")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data = pd.read_csv(args.file_path)
    data["Activity"] = data["OpenCRISPR-1 activity (day 7, %)"]
    
    test_dataset = CRISPRDataset(data)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # model directory
    models = load_models('/extdata2/YMC/OC1/saved_models/250212_1420', device)
    
    data["prediction"] = predict(models, test_loader, device)
    
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(args.save_dir, f"prediction_{timestamp}.csv")
    data.to_csv(save_path, index=False, columns=["Spacer", "Target", "OpenCRISPR-1 activity (day 7, %)", "Fold", "ontarget", "prediction"])
    
if __name__ == "__main__":
    main()