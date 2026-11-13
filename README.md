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

## Run the supplied hackathon dataset

The provided `hackathon_dataset.zip` is intentionally Git-ignored. Extract it
locally without adding or committing it:

```powershell
Expand-Archive -LiteralPath .\hackathon_dataset.zip -DestinationPath .\data\raw
```

The archive layout is `hackathon_dataset/train/{real,fake}/<image_id>.jpg`,
`hackathon_dataset/test/images/<image_id>.jpg`, plus
`train_metadata.csv` and `sample_submission.csv`. Validate that each metadata
row maps to exactly one training image and create the normalized manifest:

```powershell
python -m src.data.prepare_hackathon_data --data-root data/raw/hackathon_dataset --out-dir data/manifests
```

The provided CSV contains image IDs and labels but no video/subject IDs, so
each unique image ID is used as its own group. This keeps frames together only
if the data provides shared IDs; it cannot detect different frames from the
same person when IDs are unrelated. Use subject/video metadata if the hackathon
provides it separately.

Generate and extract the training and test crops, then create model-ready
manifests and cached frequency maps:

```powershell
python -m src.data.make_folds --data data/raw/hackathon_dataset/train --metadata data/manifests/train_metadata.csv --out data/train_folds.csv --folds 5 --seed 42
python -m src.data.extract_faces --data data/raw/hackathon_dataset/train --out data/faces/train --metadata data/train_folds.csv --margin 0.15 --no-face full-image
python -m src.data.extract_faces --data data/raw/hackathon_dataset/test/images --out data/faces/test --unlabeled --margin 0.15 --no-face full-image
python -m src.data.prepare_manifests --train-faces data/faces/train --test-faces data/faces/test --out-dir data/manifests --frequency-dir data/frequency --sample-submission data/raw/hackathon_dataset/sample_submission.csv
```

Check the `failures.csv` and `fallbacks.csv` reports after extraction. With
`--no-face full-image`, images where MTCNN does not find a confident face are
retained as full frames and listed in `fallbacks.csv`; without this option they
are skipped and listed in `failures.csv`. The test-manifest command also
verifies that all extracted test IDs match the sample submission and orders
rows to match that template. All dataset files and generated manifests, crops,
and FFT maps are Git-ignored.

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
python -m src.training.run_training --manifest data/manifests/manifest_folds.csv --folds 5 --epochs 10 --batch-size 16 --device auto
python -c "from src.training.inference_pipeline import run_ensemble_inference; run_ensemble_inference(test_manifest_csv='data/manifests/test_manifest.csv', checkpoint_dir='checkpoints', output_csv='outputs/submission.csv')"
```

The runner validates the manifest and trains all five folds, writing the
lowest-validation-loss model to `checkpoints/best_model_fold_<n>.pt` and
per-epoch metrics to `checkpoints/training_history_fold_<n>.csv`. It logs
training/validation loss, accuracy, and the learning rate. Inference accepts
both `.pt` and legacy `.pth` checkpoints and ensembles all checkpoint files in
the selected directory. Adjust `--epochs`, `--batch-size`, `--lr`, `--min-lr`,
`--num-workers`, or `--checkpoint-dir` as needed; `--device cuda` requires a
working CUDA-enabled PyTorch install. Training applies the Albumentations
pipeline to RGB images and recomputes their FFT from the augmented image so
both streams stay aligned. Validation and inference use cached frequency maps.
Both inputs are resized to 256×256 and use the same ImageNet normalization.

## Tests

Run the unit tests using the project environment:

```powershell
python -m unittest discover -s tests -v
```

The tests use temporary synthetic images and do not require the hackathon
dataset. They do not measure model accuracy or verify real-dataset group IDs.
