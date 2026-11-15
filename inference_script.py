"""
inference_script.py
===================
Standalone inference script matching official inference_template.py
for Not My Face (MTCNN Face Crop + DualStreamNet: EfficientNet-B2 + 2D-FFT).

Usage:
    python inference_script.py \
        --test_dir ./test/images/ \
        --weights ./best_model_fold_0.pt \
        --output predictions.csv
"""

import argparse
from pathlib import Path
from typing import Optional

from facenet_pytorch import MTCNN
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Matches src/data/extract_faces.py & training setup
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 256
BATCH_SIZE = 64
THRESHOLD = 0.5
FACE_MARGIN = 0.15       # Exact default --margin in src/data/extract_faces.py
MIN_FACE_CONF = 0.90     # Exact default --min-confidence in src/data/extract_faces.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTION & 2D-FFT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def build_face_detector() -> MTCNN:
    return MTCNN(keep_all=True, device=DEVICE)


def select_face_box(
    boxes: Optional[np.ndarray],
    probabilities: Optional[np.ndarray],
    min_confidence: float = MIN_FACE_CONF,
) -> Optional[tuple[float, float, float, float]]:
    """Choose the largest detected face meeting the confidence threshold."""
    if boxes is None or probabilities is None:
        return None
    valid = []
    for box, probability in zip(boxes, probabilities):
        try:
            numeric_box = np.asarray(box, dtype=np.float64)
            confidence = float(probability)
        except (TypeError, ValueError):
            continue
        if (
            numeric_box.shape == (4,)
            and np.isfinite(confidence)
            and np.all(np.isfinite(numeric_box))
            and confidence >= min_confidence
        ):
            x1, y1, x2, y2 = numeric_box.tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > 0.0:
                valid.append((area, confidence, (x1, y1, x2, y2)))
    if not valid:
        return None
    valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return valid[0][2]


def crop_primary_face(
    rgb_img: Image.Image,
    detector: Optional[MTCNN],
    margin: float = FACE_MARGIN,
    min_confidence: float = MIN_FACE_CONF,
) -> Image.Image:
    """Detect and crop the largest confident face; fall back to full image if none found."""
    if detector is None:
        return rgb_img

    try:
        boxes, probabilities = detector.detect(rgb_img)
    except Exception:
        return rgb_img

    face_box = select_face_box(boxes, probabilities, min_confidence=min_confidence)
    if face_box is None:
        return rgb_img

    x1, y1, x2, y2 = face_box
    width, height = rgb_img.size
    box_width = x2 - x1
    box_height = y2 - y1

    left = max(0, int(np.floor(x1 - box_width * margin)))
    top = max(0, int(np.floor(y1 - box_height * margin)))
    right = min(width, int(np.ceil(x2 + box_width * margin)))
    bottom = min(height, int(np.ceil(y2 + box_height * margin)))

    if right <= left or bottom <= top:
        return rgb_img

    return rgb_img.crop((left, top, right, bottom))


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
    def __init__(
        self,
        image_dir: str,
        transform=None,
        detector: Optional[MTCNN] = None,
    ):
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
        self.detector = detector

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as image:
                raw_rgb = image.convert("RGB")
        except Exception:
            raw_rgb = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))

        face_rgb = crop_primary_face(raw_rgb, self.detector)
        freq_img = compute_fft_rgb(face_rgb)

        if self.transform:
            rgb_tensor = self.transform(face_rgb)
            freq_tensor = self.transform(freq_img)
        else:
            rgb_tensor = transforms.ToTensor()(face_rgb)
            freq_tensor = transforms.ToTensor()(freq_img)

        image_id = img_path.stem
        return rgb_tensor, freq_tensor, image_id


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE — DualStreamNet (Spatial EfficientNet-B2 + 2D-FFT CNN)
# ─────────────────────────────────────────────────────────────────────────────
class FrequencyBranch(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
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

        self.spatial_backbone = backbone
        self.frequency_branch = FrequencyBranch(out_dim=256)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(spatial_dim + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        spatial_feat = self.spatial_backbone(rgb)
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
def run_inference(
    model: nn.Module,
    test_dir: str,
    batch_size: int = BATCH_SIZE,
    threshold: float = THRESHOLD,
    use_mtcnn: bool = True,
) -> pd.DataFrame:
    transform = get_inference_transforms()
    detector = build_face_detector() if use_mtcnn else None
    dataset = TestDataset(test_dir, transform=transform, detector=detector)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    all_ids = []
    all_labels = []
    all_confidences = []

    print(
        f"[INFO] Running inference on {len(dataset)} images "
        f"(batch_size={batch_size}, mtcnn={use_mtcnn})..."
    )

    with torch.inference_mode():
        for rgb_imgs, freq_imgs, batch_ids in loader:
            rgb_imgs = rgb_imgs.to(DEVICE, non_blocking=True)
            freq_imgs = freq_imgs.to(DEVICE, non_blocking=True)

            logits = model(rgb_imgs, freq_imgs)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            probs = np.clip(np.nan_to_num(probs, nan=0.5), 0.0, 1.0)
            labels = (probs >= threshold).astype(int)

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
    parser.add_argument("--no_mtcnn", action="store_true", help="Skip MTCNN if inputs are already cropped")
    args = parser.parse_args()

    model = load_model(args.weights)
    predictions = run_inference(
        model,
        args.test_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        use_mtcnn=not args.no_mtcnn,
    )

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