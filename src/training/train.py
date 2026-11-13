# src/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from ..models.dual_stream_net import DualStreamNet

class BCEWithLabelSmoothingLoss(nn.Module):
    """Binary Cross Entropy with label smoothing to prevent overconfidence."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(preds, targets)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for rgb, freq, targets in tqdm(dataloader, desc="Training"):
        rgb = rgb.to(device, non_blocking=True)
        freq = freq.to(device, non_blocking=True)
        targets = targets.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast():
            outputs = model(rgb, freq)
            loss = criterion(outputs, targets)

        # Scaled backward pass for fp16 training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * rgb.size(0)
        
        # Calculate accuracy
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct_preds += (preds == targets).sum().item()
        total_samples += rgb.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for rgb, freq, targets in tqdm(dataloader, desc="Validating"):
            rgb = rgb.to(device, non_blocking=True)
            freq = freq.to(device, non_blocking=True)
            targets = targets.float().unsqueeze(1).to(device, non_blocking=True)

            with autocast():
                outputs = model(rgb, freq)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * rgb.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_preds += (preds == targets).sum().item()
            total_samples += rgb.size(0)

    val_loss = running_loss / total_samples
    val_acc = correct_preds / total_samples
    return val_loss, val_acc


if __name__ == "__main__":
    # Smoke test / Config placeholder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DualStreamNet(num_classes=1, pretrained=True).to(device)
    print("DualStreamNet initialized successfully for training pipeline.")