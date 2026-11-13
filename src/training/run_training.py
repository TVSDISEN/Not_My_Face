"""Train and save the best validation-loss checkpoint for each data fold."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from ..data.augmentations import build_train_augmentation
from ..data.frequency import compute_fft_spectrogram, frequency_map_to_rgb
from ..models.dual_stream_net import DualStreamNet
from .train import BCEWithLabelSmoothingLoss, train_one_epoch, validate


def _create_grad_scaler(enabled):
    scaler_type = getattr(torch.amp, "GradScaler", None)
    if scaler_type is not None:
        return scaler_type("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


class HackathonDataset(Dataset):
    """Dataset wrapper for dual-stream spatial and frequency inputs."""

    def __init__(self, manifest_df, transform=None, augment=False):
        self.df = manifest_df
        self.augmentation = build_train_augmentation() if augment else None
        self.transform = transform or T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb_path = row["rgb_path"]
        freq_path = row["freq_path"]
        label = float(row.get("label", 0.0))

        with Image.open(rgb_path) as image:
            rgb_img = image.convert("RGB")
        if self.augmentation is not None:
            rgb_array = np.asarray(rgb_img)
            rgb_array = self.augmentation(image=rgb_array)["image"]
            frequency_map = compute_fft_spectrogram(rgb_array)
            frequency_rgb = frequency_map_to_rgb(frequency_map)
            freq_img = Image.fromarray(frequency_rgb)
            rgb_img = Image.fromarray(rgb_array)
        else:
            with Image.open(freq_path) as image:
                freq_img = image.convert("RGB")

        rgb_tensor = self.transform(rgb_img)
        freq_tensor = self.transform(freq_img)
        return rgb_tensor, freq_tensor, torch.tensor(label, dtype=torch.float32)


def _validate_training_manifest(
    manifest_csv: str | Path, num_folds: int
) -> pd.DataFrame:
    manifest_path = Path(manifest_csv).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    required = {"rgb_path", "freq_path", "fold", "label"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Training manifest is missing required columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    if frame.empty:
        raise ValueError(f"Training manifest is empty: {manifest_path}")

    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Training labels must be binary 0 (real) or 1 (fake).")
    if set(labels.astype(int).unique()) != {0, 1}:
        raise ValueError("Training manifest must contain both real and fake samples.")

    folds = pd.to_numeric(frame["fold"], errors="coerce")
    if folds.isna().any() or not folds.eq(folds.astype(int)).all():
        raise ValueError("Fold values must be integers.")
    if set(folds.astype(int).unique()) != set(range(num_folds)):
        raise ValueError(
            f"Training manifest must contain every fold from 0 to {num_folds - 1}."
        )

    for column in ("rgb_path", "freq_path"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Training manifest contains an empty {column}.")

    frame["label"] = labels.astype(int)
    frame["fold"] = folds.astype(int)
    return frame


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Device must be 'auto', 'cpu', or 'cuda'.")
    return device


def run_kfold_training(
    manifest_csv="data/manifests/manifest_folds.csv",
    num_folds=5,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    num_workers=2,
    checkpoint_dir="checkpoints",
    device="auto",
    pretrained=True,
    min_lr=1e-6,
):
    """Train all folds and save each fold's lowest-validation-loss model."""
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("epochs and batch_size must be positive; num_workers cannot be negative.")
    if lr <= 0 or min_lr < 0 or min_lr > lr:
        raise ValueError("Learning rates must satisfy 0 <= min_lr <= lr and lr > 0.")

    selected_device = _select_device(device)
    print(f"Starting {num_folds}-fold training on {selected_device}.")
    full_df = _validate_training_manifest(manifest_csv, num_folds)
    checkpoint_root = Path(checkpoint_dir).expanduser()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    criterion = BCEWithLabelSmoothingLoss(smoothing=0.1)
    summaries = []

    for fold in range(num_folds):
        print(f"\n--- Training fold {fold + 1}/{num_folds} ---")
        train_df = full_df[full_df["fold"] != fold].reset_index(drop=True)
        val_df = full_df[full_df["fold"] == fold].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError(f"Fold {fold} has an empty train or validation split.")
        if set(train_df["label"].unique()) != {0, 1}:
            raise ValueError(f"Training split for fold {fold} must contain both classes.")
        if set(val_df["label"].unique()) != {0, 1}:
            raise ValueError(f"Validation split for fold {fold} must contain both classes.")

        train_dataset = HackathonDataset(train_df, augment=True)
        val_dataset = HackathonDataset(val_df)
        loader_options = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": selected_device.type == "cuda",
        }
        train_loader = DataLoader(
            train_dataset, shuffle=True, **loader_options
        )
        val_loader = DataLoader(
            val_dataset, shuffle=False, **loader_options
        )

        model = DualStreamNet(num_classes=1, pretrained=pretrained).to(selected_device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        scaler = _create_grad_scaler(enabled=selected_device.type == "cuda")
        best_val_loss = float("inf")
        best_summary = None
        history = []
        checkpoint_path = checkpoint_root / f"best_model_fold_{fold}.pt"
        history_path = checkpoint_root / f"training_history_fold_{fold}.csv"

        for epoch in range(1, epochs + 1):
            epoch_lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, selected_device
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion, selected_device
            )
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": epoch_lr,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
            )
            pd.DataFrame(history).to_csv(history_path, index=False)
            print(
                f"Epoch {epoch}/{epochs} | lr {epoch_lr:.2e} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_summary = {
                    "fold": fold,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                torch.save(
                    {
                        **best_summary,
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )
                print(f"Saved best checkpoint: {checkpoint_path}")

            scheduler.step()

        summaries.append(best_summary)
        print(
            f"Best fold {fold} checkpoint: epoch {best_summary['epoch']}, "
            f"validation loss {best_summary['val_loss']:.4f}, "
            f"accuracy {best_summary['val_acc']:.4f}"
        )

    print(f"\nCompleted {num_folds}-fold training. Checkpoints: {checkpoint_root}")
    return summaries


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/manifests/manifest_folds.csv",
        help="Training manifest containing rgb_path, freq_path, fold, and label.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_kfold_training(
        manifest_csv=args.manifest,
        num_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        pretrained=not args.no_pretrained,
    )
