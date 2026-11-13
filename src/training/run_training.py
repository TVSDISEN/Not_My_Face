# src/training/run_training.py
import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as T

from ..models.dual_stream_net import DualStreamNet
from .train import BCEWithLabelSmoothingLoss, train_one_epoch, validate

class HackathonDataset(Dataset):
    """Dataset wrapper for Dual-Stream Spatial (RGB) and Frequency inputs."""
    def __init__(self, manifest_df, transform=None):
        self.df = manifest_df
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb_path = row['rgb_path']
        freq_path = row['freq_path']
        label = float(row['label'])

        # Load RGB image
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_tensor = self.transform(rgb_img)

        # Load Frequency spectrum image
        freq_img = Image.open(freq_path).convert('RGB')
        freq_tensor = self.transform(freq_img)

        return rgb_tensor, freq_tensor, torch.tensor(label, dtype=torch.float32)


def run_kfold_training(manifest_csv="manifest_folds.csv", num_folds=5, epochs=10, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting 5-Fold Training Pipeline on device: {device}")

    if not os.path.exists(manifest_csv):
        print(f"⚠️ Manifest file {manifest_csv} not found! Please ensure Member 1's fold manifest is generated.")
        return

    full_df = pd.read_csv(manifest_csv)
    criterion = BCEWithLabelSmoothingLoss(smoothing=0.1)
    os.makedirs("checkpoints", exist_ok=True)

    for fold in range(num_folds):
        print(f"\n--- Training Fold {fold + 1}/{num_folds} ---")
        
        train_df = full_df[full_df['fold'] != fold].reset_index(drop=True)
        val_df = full_df[full_df['fold'] == fold].reset_index(drop=True)

        train_dataset = HackathonDataset(train_df)
        val_dataset = HackathonDataset(val_df)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = DualStreamNet(num_classes=1, pretrained=True).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=1e-6)
        scaler = GradScaler()

        best_val_acc = 0.0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'fold': fold,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc
                }, f"checkpoints/best_model_fold_{fold}.pth")
                print(f"💾 Saved new best model for fold {fold} (Val Acc: {val_acc:.4f})")

    print("✅ 5-Fold Training Pipeline completed successfully!")

if __name__ == "__main__":
    run_kfold_training()