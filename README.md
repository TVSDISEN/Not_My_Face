# NotMyFace — deepfake detection

NotMyFace is a dual-stream image detector for the Deepfake Detection Hackathon.
The spatial stream reads RGB face crops; the frequency stream reads a
three-channel PNG encoding of the normalized 2D FFT log magnitude. Training and
inference consume the manifests described below.

## Setup

Use Python 3.10–3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put only the provided dataset under `data/raw/`. Do not add external training
data. Supported image formats are JPG, JPEG, PNG, BMP, and WebP.

## Prepare training and test data

### 1. Build grouped training folds

The auto-discovered training layout is class directory → video/subject
directory → frames, for example:

```text
data/raw/train/real/video_001/frame_001.jpg
data/raw/train/fake/video_002/frame_001.jpg
```

Real directories are named `real`, `authentic`, or `original`; fake directories
are named `fake`, `deepfake`, or `synthetic` (numeric class directories `0` and
`1` are also accepted). In this layout, the first directory below the class
directory becomes the group ID. Check that this is the real video/subject
identity before using the inferred grouping. For other layouts, give
`make_folds` an explicit CSV with `path,label,group_id` columns; paths are
relative to the raw data directory and labels are `real`/`fake` or `0`/`1`.

```powershell
python -m src.data.make_folds --data data/raw/train --out data/train_folds.csv --folds 5 --seed 42
# Or supply an explicit mapping:
python -m src.data.make_folds --data data/raw/train --metadata data/train_metadata.csv --out data/train_folds.csv
```

Each group must appear in only one fold. Five folds need at least five distinct
groups per class. The output has `path,label,group_id,fold`.

### 2. Extract training and test face crops

Pass the fold CSV when extracting training crops so the split remains attached
to every crop. For test images, use `--unlabeled`; this scans the image tree and
preserves each relative input name as `filename`. Test crops have no labels.

```powershell
python -m src.data.extract_faces --data data/raw/train --out data/faces/train --metadata data/train_folds.csv --margin 0.15
python -m src.data.extract_faces --data data/raw/test --out data/faces/test --unlabeled --margin 0.15
```

MTCNN uses CUDA when available and otherwise CPU. It keeps the largest face
detected with confidence at least 0.90 and expands its box by 15%. Images
without a confident face are listed in `failures.csv` and skipped by default.
Use `--no-face full-image` only if a full frame is an acceptable fallback.
Unreadable images and failed writes are reported and make extraction exit
non-zero. Check the report before proceeding so skipped data is understood.

### 3. Generate model-ready manifests and FFT maps

```powershell
python -m src.data.prepare_manifests --train-faces data/faces/train --test-faces data/faces/test --out-dir data/manifests --frequency-dir data/frequency
```

The command validates every referenced crop, validates binary training labels
and folds, checks that groups do not cross folds, and writes one frequency PNG
per crop. Outputs:

- `data/manifests/manifest_folds.csv` with exactly `rgb_path,freq_path,fold,label`.
- `data/manifests/test_manifest.csv` with `rgb_path,freq_path,filename`, as
  expected by the ensemble inference pipeline. Paths in both manifests are
  absolute for compatibility with `HackathonDataset`; `filename` retains the
  test image's relative path and is consumed by inference for submission rows.
- `data/frequency/` containing cached frequency PNG files. The FFT map is
  normalized to `[0,1]`, quantized to 8-bit, and replicated across RGB channels
  to match the model's three-channel input.

If test crops were produced without the `--unlabeled` option, the manifest
builder falls back to `original_path` and then the crop path for `filename`.
It does not invent a real test label.

### 4. Train and infer

```powershell
python -c "from src.training.run_training import run_kfold_training; run_kfold_training(manifest_csv='data/manifests/manifest_folds.csv', num_folds=5)"
python -c "from src.training.inference_pipeline import run_ensemble_inference; run_ensemble_inference(test_manifest_csv='data/manifests/test_manifest.csv', checkpoint_dir='checkpoints', output_csv='outputs/submission.csv')"
```

The training runner and inference pipeline can also be run with their defaults
when the manifest files are in the current directory. Review their options
before a long GPU run. Training applies the Albumentations pipeline to RGB
images and recomputes their FFT from the augmented image so both streams stay
aligned. Validation and inference use the cached frequency maps. Both inputs
are resized to 256×256 and use the same ImageNet normalization.

## Tests

Run the unit tests using the project environment:

```powershell
python -m unittest discover -s tests -v
```

The tests use temporary synthetic images and do not require the hackathon
dataset. They do not measure model accuracy or verify real-dataset group IDs.
