# NotMyFace — Dual-Stream Deepfake Detection

**NotMyFace** is a lightweight, dual-stream deepfake image detector built for the Deepfake Detection Hackathon. It combines a **spatial RGB stream (`EfficientNet-B2`)** with a **frequency stream (`2D-FFT` log-amplitude spectrum CNN)** to capture both facial blending inconsistencies and generative upsampling artifacts.

---

## Architecture Overview

```text
Input Face Crop (256 × 256 × 3)
 ├──► Spatial Branch (EfficientNet-B2) ──────► 1,408-d features ──┐
 └──► 2D-FFT Spectrum ──► Frequency CNN ─────►   256-d features ──┴──► Concat (1,664-d) ──► MLP Head ──► P(Fake)
```

* **Spatial Stream:** `EfficientNet-B2` extracts fine-grained visual texture and boundary artifacts into a `1,408`-dimensional representation.
* **Frequency Stream:** Computes a centered 2D-FFT log-magnitude map normalized to `[0, 1]`, quantized to 8-bit, and replicated across 3 channels (`256 × 256 × 3`), encoded via a 3-layer strided CNN into `256` features.
* **Fusion & Classifier Head:** Concatenates both streams (`1,664`-d) into `Dropout(0.4) -> Linear(1664, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 1)`.
* **Model Footprint:** **8,680,707** parameters (~8.68M) | **33.58 MB** (PyTorch `.pt`) | **33.11 MB** (ONNX `.onnx`).

---

## Team Contributions

* **Abhinav (Data Engineering & Preprocessing):** MTCNN face detection and alignment (`extract_faces.py`), 2D-FFT spectrogram generation (`frequency.py`), dataset validation (`prepare_hackathon_data.py`), manifest generation (`prepare_manifests.py`), and grouped fold splitting (`make_folds.py`).
* **Disen (Model Architecture & Training Pipeline):** `DualStreamNet` architecture (`dual_stream_net.py`), Albumentations augmentation pipeline (`augmentations.py`), and 5-fold stratified cross-validation training with label smoothing (`run_training.py`, `train.py`).
* **Siddharth (Inference, Optimization & Benchmarking):** Multi-fold ensemble inference with Test-Time Augmentation and ONNX Runtime acceleration (`inference_pipeline.py`, `predict.py`), standalone submission script (`inference_script.py`), Opset 17 ONNX exporter (`export_onnx.py`), latency/throughput profiler (`benchmark.py`), and automated test suite (`tests/`).

---

## Setup & Installation

Use **Python 3.10–3.12**. From the repository root:

### macOS / Linux (`bash` / `zsh`)
```bash
git clone https://github.com/TVSDISEN/Not_My_Face.git
cd Not_My_Face

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (`PowerShell`)
```powershell
git clone https://github.com/TVSDISEN/Not_My_Face.git
cd Not_My_Face

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*Supported image formats: `JPG`, `JPEG`, `PNG`, `BMP`, and `WebP`. Place only the provided dataset under `data/raw/` (all raw datasets, extracted crops, FFT maps, and checkpoints are Git-ignored).*

---

## Quickstart: Run the Hackathon Dataset

### 1. Extract & Validate the Supplied Archive
Unpack `hackathon_dataset.zip` locally into `data/raw/` without committing it:

```bash
# macOS / Linux
unzip hackathon_dataset.zip -d data/raw
```
```powershell
# Windows PowerShell
Expand-Archive -LiteralPath .\hackathon_dataset.zip -DestinationPath .\data\raw
```

Validate that every row in `train_metadata.csv` maps to a unique training image (`hackathon_dataset/train/{real,fake}/<image_id>.jpg` and `hackathon_dataset/test/images/<image_id>.jpg`) and generate the normalized metadata manifest:

```bash
python -m src.data.prepare_hackathon_data \
  --data-root data/raw/hackathon_dataset \
  --out-dir data/manifests
```

*(Note: Because `train_metadata.csv` provides image IDs and labels without video/subject IDs, each unique image ID acts as its own group. If subject/video IDs are provided separately, supply them via `--group-column`.)*

### 2. Create Stratified Folds, Extract Face Crops & Cache FFT Maps
```bash
# 1. Create 5-fold stratified split
python -m src.data.make_folds \
  --data data/raw/hackathon_dataset/train \
  --metadata data/manifests/train_metadata.csv \
  --out data/train_folds.csv \
  --folds 5 \
  --seed 42

# 2. Extract training face crops (largest face with confidence >= 0.90, 15% margin, full-image fallback)
python -m src.data.extract_faces \
  --data data/raw/hackathon_dataset/train \
  --out data/faces/train \
  --metadata data/train_folds.csv \
  --margin 0.15 \
  --no-face full-image

# 3. Extract unlabeled test face crops
python -m src.data.extract_faces \
  --data data/raw/hackathon_dataset/test/images \
  --out data/faces/test \
  --unlabeled \
  --margin 0.15 \
  --no-face full-image

# 4. Generate model-ready manifests and 3-channel 2D-FFT PNG maps aligned to sample_submission.csv
python -m src.data.prepare_manifests \
  --train-faces data/faces/train \
  --test-faces data/faces/test \
  --out-dir data/manifests \
  --frequency-dir data/frequency \
  --sample-submission data/raw/hackathon_dataset/sample_submission.csv
```

> **Extraction Audit Tip:** Inspect `failures.csv` and `fallbacks.csv` inside `data/faces/{train,test}/` after extraction. With `--no-face full-image`, images without a confident MTCNN detection (`>= 0.90`) are retained as full frames in `fallbacks.csv` rather than dropped. `prepare_manifests` also verifies that test IDs match `sample_submission.csv` 1-to-1.

---

## Custom Dataset Layout (Optional)

If running on a custom directory tree organized as `data/raw/train/{real,fake}/<group_id>/frame_001.jpg` (accepts `real`/`authentic`/`original`/`0` and `fake`/`deepfake`/`synthetic`/`1`):

```bash
# Auto-discover group IDs from subdirectories (or pass --metadata data/train_metadata.csv)
python -m src.data.make_folds --data data/raw/train --out data/train_folds.csv --folds 5 --seed 42

# Extract crops (omit --no-face full-image to skip frames without confident faces)
python -m src.data.extract_faces --data data/raw/train --out data/faces/train --metadata data/train_folds.csv --margin 0.15
python -m src.data.extract_faces --data data/raw/test --out data/faces/test --unlabeled --margin 0.15

# Build manifests & cached FFT maps
python -m src.data.prepare_manifests --train-faces data/faces/train --test-faces data/faces/test --out-dir data/manifests --frequency-dir data/frequency
```

**Generated Manifest Schemas:**
* `data/manifests/manifest_folds.csv`: `rgb_path,freq_path,fold,label`
* `data/manifests/test_manifest.csv`: `rgb_path,freq_path,filename`

---

## Training, Optimization & Inference

### 1. Train 5-Fold Cross-Validation
Trains all 5 folds with `AdamW` (`lr = 1e-4`), `CosineAnnealingLR`, BCE with label smoothing ($\epsilon = 0.1$), and on-the-fly Albumentations + recomputed FFT augmentations. Writes `checkpoints/best_model_fold_<n>.pt` and `checkpoints/training_history_fold_<n>.csv`:

```bash
python -m src.training.run_training \
  --manifest data/manifests/manifest_folds.csv \
  --folds 5 \
  --epochs 10 \
  --batch-size 16 \
  --device auto
```

### 2. Export to ONNX & Profile Latency
Exports PyTorch checkpoints to ONNX (Opset 17, dynamic batch axes) with numerical parity verification (`max diff <= 1e-4`) and profiles CPU/GPU latency and throughput:

```bash
# Export checkpoints to ONNX
python -m src.training.export_onnx \
  --checkpoint-dir checkpoints \
  --output-dir onnx_models

# Benchmark PyTorch vs. ONNX Runtime
python -m src.training.benchmark \
  --checkpoint checkpoints/best_model_fold_0.pt \
  --onnx onnx_models/best_model_fold_0.onnx \
  --warmup 3 \
  --runs 15 \
  --output-json benchmarks/efficiency_report.json
```

### 3. Generate Predictions (`predictions.csv`)

#### Option A: Standalone Inference Script (Official Submission Interface)
Runs directly on a flat directory of raw or cropped test images using a single `.pt` weight file (handles MTCNN cropping, 2D-FFT computation, and `image_id,label,confidence` formatting automatically):

```bash
python inference_script.py \
  --test_dir data/raw/hackathon_dataset/test/images \
  --weights checkpoints/best_model_fold_0.pt \
  --output predictions.csv
```

#### Option B: Multi-Fold Ensemble CLI (`src.training.predict`)
Ensembles all trained fold checkpoints with horizontal-flip Test-Time Augmentation (TTA) and aligns output rows to `sample_submission.csv`:

```bash
# PyTorch 5-fold ensemble with TTA
python -m src.training.predict \
  --test-manifest data/manifests/test_manifest.csv \
  --checkpoint-dir checkpoints \
  --sample-submission data/raw/hackathon_dataset/sample_submission.csv \
  --output-csv predictions.csv

# Accelerated ONNX Runtime 5-fold ensemble (7.55x faster on CPU)
python -m src.training.predict \
  --test-manifest data/manifests/test_manifest.csv \
  --use-onnx \
  --onnx-dir onnx_models \
  --no-tta \
  --sample-submission data/raw/hackathon_dataset/sample_submission.csv \
  --output-csv predictions.csv
```

---

## Automated Tests

Run the test suite (uses temporary synthetic images; does not require the hackathon dataset):

```bash
# Using pytest (recommended)
pytest -v

# Or using standard library unittest
python -m unittest discover -s tests -v
```

---

## Results & Efficiency Benchmarks

### 5-Fold Cross-Validation Performance

| Fold | Best Epoch | Best Val Loss | Best Val Accuracy |
| :--- | :---: | :---: | :---: |
| **Fold 0** | 8 | 0.4913 | **84.36%** |
| **Fold 1** | 4 | 0.4967 | **82.43%** |
| **Fold 2** | 7 | 0.4946 | **83.07%** |
| **Fold 3** | 3 | 0.5055 | **81.00%** |
| **Fold 4** | 6 | **0.4840** | **84.00%** |
| **5-Fold Mean** | — | **0.4944** | **82.97%** |

### CPU Inference Benchmarks (`DualStreamNet`, `3 × 256 × 256`)

| Batch Size | Runtime | Artifact Size | Mean Latency | P95 Batch Latency | Throughput | Speedup |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | PyTorch (CPU) | 33.58 MB | 77.28 ms/img | 93.47 ms | 12.94 img/s | 1.00x |
| **1** | **ONNX Runtime (CPU)** | **33.11 MB** | **10.23 ms/img** | **10.73 ms** | **97.71 img/s** | **7.55x** |
| **8** | PyTorch (CPU) | 33.58 MB | 32.92 ms/img | 270.19 ms | 30.38 img/s | 1.00x |
| **8** | **ONNX Runtime (CPU)** | **33.11 MB** | **12.31 ms/img** | **99.38 ms** | **81.21 img/s** | **2.67x** |