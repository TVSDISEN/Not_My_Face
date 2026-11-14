"""
inference_script.py
===================
Standalone inference script matching official inference_template.py
for Not My Face (DualStreamNet: EfficientNet-B2 + 2D-FFT Frequency Branch).

Usage:
    python inference_script.py \
        --test_dir ./test/images/ \
        --weights ./best_model_fold_0.pt \
        --output predictions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 4
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 2D-FFT FREQUENCY EXTRACTION (Matches src/data/frequency.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fft_rgb(rgb_img: Image.Image) -> Image.Image:
    """Compute centered 2D-FFT log-amplitude spectrum and map to 3-channel RGB."""
    gray = np.asarray(rgb_img.convert("L"), dtype=np.float32)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    mag_min, mag_max = float(magnitude.min()), float(magnitude.max())
    if mag_max > mag_min:
        norm = ((magnitude - mag_min) / (mag_max - mag_min) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(magnitude, dtype=np.uint8)

    freq_rgb = np.stack([norm, norm, norm], axis=-1)
    return Image.fromarray(freq_rgb)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — Reads test images from a flat directory
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Test directory not found: {self.image_dir}")

        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg"))
            + list(self.image_dir.glob("*.jpeg"))
            + list(self.image_dir.glob("*.png"))
            + list(self.image_dir.glob("*.JPG"))
            + list(self.image_dir.glob("*.JPEG"))
            + list(self.image_dir.glob("*.PNG"))
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as image:
            rgb_img = image.convert("RGB")
        freq_img = compute_fft_rgb(rgb_img)

        if self.transform:
            rgb_tensor = self.transform(rgb_img)
            freq_tensor = self.transform(freq_img)
        else:
            rgb_tensor = transforms.ToTensor()(rgb_img)
            freq_tensor = transforms.ToTensor()(freq_img)

        image_id = img_path.stem  # Filename without extension
        return rgb_tensor, freq_tensor, image_id


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE — DualStreamNet (Spatial EfficientNet-B2 + 2D-FFT CNN)
# ─────────────────────────────────────────────────────────────────────────────
class FrequencyBranch(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DualStreamNet(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = False):
        super().__init__()
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
        spatial_dim = backbone.classifier[1].in_features  # 1408
        backbone.classifier = nn.Identity()

        self.spatial_branch = backbone
        self.frequency_branch = FrequencyBranch(out_dim=256)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(spatial_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.spatial_branch(rgb)
        freq_feat = self.frequency_branch(freq)
        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(fused)


def build_model() -> nn.Module:
    try:
        from src.models.dual_stream_net import DualStreamNet as RepoDualStreamNet
        return RepoDualStreamNet(num_classes=1, pretrained=False)
    except ImportError:
        return DualStreamNet(num_classes=1, pretrained=False)


def load_model(weights_path: str) -> nn.Module:
    model = build_model()
    state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Model loaded from {weights_path} on {DEVICE}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model: nn.Module, test_dir: str) -> pd.DataFrame:
    transform = get_inference_transforms()
    dataset = TestDataset(test_dir, transform=transform)
    workers = NUM_WORKERS if DEVICE == "cuda" else 0
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=(DEVICE == "cuda"),
    )

    all_ids = []
    all_labels = []
    all_confidences = []

    print(f"[INFO] Running inference on {len(dataset)} images (batch_size={BATCH_SIZE})...")

    with torch.inference_mode():
        for rgb_imgs, freq_imgs, batch_ids in loader:
            rgb_imgs = rgb_imgs.to(DEVICE, non_blocking=True)
            freq_imgs = freq_imgs.to(DEVICE, non_blocking=True)

            logits = model(rgb_imgs, freq_imgs)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            probs = np.clip(np.nan_to_num(probs, nan=0.5), 0.0, 1.0)
            labels = (probs >= THRESHOLD).astype(int)

            all_ids.extend(list(batch_ids))
            all_labels.extend(labels.tolist())
            all_confidences.extend(probs.tolist())

    return pd.DataFrame({
        "image_id": all_ids,
        "label": all_labels,
        "confidence": all_confidences,
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake ML Hackathon — Inference Script")
    parser.add_argument("--test_dir", required=True, help="Path to test images directory")
    parser.add_argument("--weights", required=True, help="Path to model weights file")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    global BATCH_SIZE, THRESHOLD
    BATCH_SIZE = args.batch_size
    THRESHOLD = args.threshold

    model = load_model(args.weights)
    predictions = run_inference(model, args.test_dir)

    # Validate output per template requirements
    assert "image_id" in predictions.columns, "Missing image_id column"
    assert "label" in predictions.columns, "Missing label column"
    assert "confidence" in predictions.columns, "Missing confidence column"
    assert predictions["confidence"].between(0, 1).all(), "Confidence values must be in [0, 1]"
    assert predictions["label"].isin([0, 1]).all(), "Labels must be 0 or 1"

    predictions.to_csv(args.output, index=False)
    print(f"[DONE] Predictions saved to {args.output}")
    print(f"  Total images: {len(predictions)}")
    print(f"  Predicted FAKE: {predictions['label'].sum()} ({predictions['label'].mean()*100:.1f}%)")
    print(f"  Predicted REAL: {(predictions['label']==0).sum()}")
    print(f"  Mean confidence: {predictions['confidence'].mean():.4f}")


if __name__ == "__main__":
    main()