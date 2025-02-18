import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class CRISPRDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Store the original indices
        self.indices = list(data.index)
        # Create position-based encoded sequences
        self.encoded_sequences = [
            (self._one_hot_encode(row['Spacer'], pad_left=False),
             self._one_hot_encode(row['Target'], pad_left=True))
            for _, row in data.iterrows()
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spacer, target = self.encoded_sequences[idx]
        # Use the stored indices to get the correct row
        actual_idx = self.indices[idx]
        row = self.data.loc[actual_idx]
        activity = np.log1p(row['Activity'])
        ontarget = row['ontarget']
        
        return (torch.tensor(spacer, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(activity, dtype=torch.float32),
                torch.tensor(ontarget, dtype=int))

    def _one_hot_encode(self, seq, length=31, pad_left=True):
        bases = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        encoding = np.zeros((4, length))
        seq_len = len(seq)
        start_idx = length - seq_len if pad_left else (length - seq_len) // 2
        for i, base in enumerate(seq):
            encoding[bases[base], start_idx + i] = 1
        return encoding

class CRISPRNet(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(CRISPRNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        self.dense = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 7, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, spacer, target):
        x = torch.cat((spacer, target), dim=1) # b x 8 x 31
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        
        return x
    
class BalancedMSELoss(nn.Module):
    def __init__(self):
        super(BalancedMSELoss, self).__init__()

        self.factor = [0.5, 1.0]  # Scaling factors for different datasets

        # Choose between standard MSE loss and scaled MSE loss
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, actual, ontarget):
        pred = pred.view(-1, 1)
        y = actual.view(-1, 1)

        # Compute loss for each dataset type
        l1 = self.mse(pred[ontarget == 1], y[ontarget == 1]) * self.factor[0]
        l2 = self.mse(pred[ontarget != 1], y[ontarget != 1]) * self.factor[1]

        # Combine losses and normalize by the batch size
        return (l1 + l2) / pred.size(0)