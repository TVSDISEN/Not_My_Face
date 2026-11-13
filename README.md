# 🙅 NotMyFace: Deepfake Detection

> *"That's not my face."* Every deepfake, caught.

**NotMyFace** is a lightweight, robust dual-stream (Spatial + Frequency) deepfake detector built for the **Deepfake Detection Hackathon**. It combines an RGB spatial stream with an FFT frequency stream, is trained only on the provided dataset, and is exported to ONNX/TensorRT for fast inference.

> **Scoring focus:** 50% Accuracy · 20% Efficiency · 20% Robustness · 10% Documentation

---

## 📌 Table of Contents
1. [Overview](#-overview)
2. [Pipeline](#-pipeline)
3. [Model Architecture](#-model-architecture)
4. [Repository Structure](#-repository-structure)
5. [Setup](#-setup)
6. [Usage](#-usage)
7. [Results](#-results)
8. [Efficiency Benchmarks](#-efficiency-benchmarks)
9. [Robustness](#-robustness)
10. [Reproducibility](#-reproducibility)
11. [Team](#-team)
12. [Rules Compliance](#-rules-compliance)
13. [License](#-license)

---

## 🔍 Overview

Deepfakes alter the face, not the background. Our approach:

- **Crops faces** to remove irrelevant background noise.
- **Stress-tests the model** with JPEG compression, blur and noise so it holds up on low-quality social media video.
- **Uses two views of each face:** visible flaws (spatial) and hidden generator fingerprints (frequency).
- **Optimizes for deployment** with a small backbone and ONNX/TensorRT export.

Output: a confidence score in `[0, 1]` per sample (`0` = Real, `1` = Fake), written to `submission.csv`.

---

## 🔄 Pipeline

```
Input Data → Face Extraction → Preprocessing + Augmentation → Dual-Stream Features
          → Feature Fusion & Frame Pooling → Classifier → submission.csv
```

| Step | Description |
|------|-------------|
| 1. Input | Raw images/videos from the provided dataset |
| 2. Face extraction | RetinaFace / MTCNN, crop with 15% margin |
| 3. Preprocessing | Uniform resize; JPEG compression, motion/Gaussian blur, color noise (train only) |
| 4. Dual-stream | Spatial (RGB) and Frequency (FFT/DCT) branches |
| 5. Fusion & pooling | Concatenate stream features, aggregate across video frames |
| 6. Classifier | Linear head → sigmoid score in `[0, 1]` |
| 7. Output | `submission.csv` in the required format |

---

## 🧠 Model Architecture

- **Spatial stream:** EfficientNet-B2/B3 (or Swin-Tiny) for skin texture, eye symmetry and blending boundaries.
- **Frequency stream:** 2D-FFT / DCT spectrum of the face crop, fed to a small CNN to catch pixel-level generator artifacts.
- **Fusion:** feature concatenation followed by temporal (frame-level) pooling.
- **Training:** mixed precision (fp16), BCE with label smoothing, cosine LR schedule.
- **Validation:** 5-fold Group K-Fold, grouped by video/subject ID to prevent leakage.
- **Inference:** horizontal-flip TTA, ONNX / TensorRT export.

---

## 📁 Repository Structure

```
NotMyFace/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/                      
│   ├── raw/
│   ├── faces/
│   └── folds.csv
├── src/
│   ├── data/
│   │   ├── make_folds.py
│   │   ├── extract_faces.py
│   │   ├── augmentations.py
│   │   ├── frequency.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── spatial_stream.py
│   │   ├── frequency_stream.py
│   │   └── dual_stream.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── benchmark.py
│   └── predict.py
├── scripts/
│   ├── run_all.sh
│   └── prepare_data.sh
├── notebooks/
│   └── error_analysis.ipynb
├── weights/                   
├── outputs/
│   └── submission.csv
├── docs/
│   ├── report.pdf
│   ├── flow_diagram.png
│   └── demo_video_link.md
└── tests/
    └── test_smoke.py
```

---

## 🔧 Setup

```bash
git clone https://github.com/ybsid/NotMyFace.git
cd NotMyFace

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, torchvision, timm, albumentations, opencv-python, facenet-pytorch (MTCNN) or retinaface, onnx, onnxruntime-gpu, scikit-learn, pandas, numpy.

Place the provided dataset in `data/raw/`. Do **not** add any external data.

---

## 🚀 Usage

```bash
# 1. Build leak-free 5-fold splits (grouped by video/subject ID)
python src/data/make_folds.py --data data/raw --out data/folds.csv --folds 5

# 2. Extract and cache face crops (15% margin)
python src/data/extract_faces.py --data data/raw --out data/faces --margin 0.15

# 3. Train (one fold, or all folds)
python src/train.py --config configs/default.yaml --fold 0

# 4. Evaluate
python src/evaluate.py --config configs/default.yaml --weights weights/best.pth

# 5. Export to ONNX
python src/export_onnx.py --weights weights/best.pth --out weights/model.onnx

# 6. Benchmark latency and memory
python src/benchmark.py --model weights/model.onnx

# 7. Generate submission
python src/predict.py --input <test_dir> --weights weights/model.onnx --out outputs/submission.csv
```

Run everything end to end:

```bash
bash scripts/run_all.sh
```

---

## 📊 Results

| Model | CV AUC | CV F1 | Notes |
|-------|--------|-------|-------|
| Baseline (spatial only) | TBD | TBD | |
| Dual-stream | TBD | TBD | |
| Dual-stream + TTA | TBD | TBD | Final |

---

## ⚡ Efficiency Benchmarks

| Format | Latency (ms/img) | Peak Memory (MB) | Params |
|--------|------------------|------------------|--------|
| PyTorch fp32 | TBD | TBD | TBD |
| PyTorch fp16 | TBD | TBD | TBD |
| ONNX | TBD | TBD | TBD |
| TensorRT | TBD | TBD | TBD |

Hardware used: _TBD_

---

## 💪 Robustness

Evaluated on clean data and on degraded versions (JPEG quality, blur, noise):

| Condition | AUC |
|-----------|-----|
| Clean | TBD |
| JPEG (Q=30) | TBD |
| Gaussian blur | TBD |
| Motion blur | TBD |
| Gaussian noise | TBD |

---

## 🔁 Reproducibility

- Fixed random seeds (set in `configs/default.yaml`).
- Fold assignments saved to `data/folds.csv`.
- All hyperparameters live in `configs/`.
- Final weights: `weights/` (or the Drive link: _TBD_, set to "Anyone with the link can view").

---

## 👥 Team

| Member | Role | Focus |
|--------|------|-------|
| Abhinav | Data & Robustness Lead | Splits, face cropping, augmentations, FFT helper |
| Disen | Model Architect & Training Lead | Architecture, training, tuning, error analysis |
| Siddharth | Optimization, Inference & Docs Lead | ONNX/TensorRT, inference, submission CSV, docs |

---

## ✅ Rules Compliance

- Trained **only** on the provided custom dataset (no external data).
- Validation is grouped by video ID to avoid leakage.
- Submission follows the required `submission.csv` format.

---

## 📄 License

MIT License. See [LICENSE](LICENSE).
