# NotMyFace — deepfake detection data foundation

NotMyFace is a dual-stream deepfake-detection hackathon project. This first
implementation establishes the data foundation: manifest creation, leak-aware
grouped folds, offline face crops, frequency transforms, and training-only
image augmentations. It does not include the classifier or training loop yet.

## Setup

Use Python 3.10–3.12 (the MTCNN dependency pins a PyTorch version in that
range). From the repository root, create and activate a virtual environment,
then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Put only the provided hackathon data under `data/raw/`. Do not add external
training data. The expected image extensions are JPG, JPEG, PNG, BMP, and WebP.
Generated crops, fold files, virtual environments, and model outputs are ignored
by Git.

## Data format and split safety

The easiest automatically detected layout is one class directory followed by
one directory per video or subject, with frames inside:

```text
data/raw/
  real/
    interview_001/frame_0001.jpg
    interview_001/frame_0002.jpg
  fake/
    generated_001/frame_0001.jpg
```

Real directories are named `real`, `authentic`, or `original`; fake directories
are named `fake`, `deepfake`, or `synthetic` (numeric class directories `0` and
`1` are also accepted). In this layout, the first directory below the class
directory is used as the group ID. The inferred group is only a convenience:
for flat image directories or any layout where that directory is not the true
video/subject identity, provide an explicit CSV manifest instead.

The manifest CSV must have `path`, `label`, and `group_id` columns. Paths are
relative to `data/raw/` (absolute paths inside that directory also work);
labels may be `real`/`fake` or `0`/`1` (`0` = real, `1` = fake). `group_id` must
identify the video or subject, and the same ID must be used for every related
frame, including any across-class pairs.

```csv
path,label,group_id
frames/a_001.jpg,real,subject_a
frames/a_002.jpg,real,subject_a
frames/b_001.jpg,fake,subject_b
```

Build the five stratified group folds:

```powershell
python -m src.data.make_folds --data data/raw --out data/folds.csv --folds 5 --seed 42
# Or use explicit labels and groups:
python -m src.data.make_folds --data data/raw --metadata data/manifest.csv --out data/folds.csv
```

The output contains `path`, normalized binary `label`, `group_id`, and `fold`.
Every input frame is assigned to one validation fold, and each group appears in
exactly one fold. The command stops with an actionable error if the requested
number of folds cannot be supported by the number of groups in each class.

## Face extraction

Extract one largest/high-confidence face crop per frame, expanded by 15% on
each side. MTCNN uses CUDA when available and otherwise runs on CPU:

```powershell
python -m src.data.extract_faces --data data/raw --out data/faces --metadata data/folds.csv --margin 0.15
```

The output mirrors the input relative paths and includes a `manifest.csv` with
the source label, group, fold (if present), and `original_path`. Images where no
face is detected are recorded in `failures.csv` and skipped by default. To
explicitly keep a full-frame image when no face is found, pass
`--no-face full-image`. Unreadable images or failed writes are reported and make
the command exit non-zero after the run. Face extraction requires the
`facenet-pytorch` MTCNN dependency from `requirements.txt`.

## Frequency stream and training augmentations

The FFT helper accepts a grayscale or RGB NumPy image and returns a normalized
2D log-magnitude spectrum as `float32`:

```python
from src.data.frequency import compute_fft_spectrogram

frequency_map = compute_fft_spectrogram(rgb_image)
```

`src.data.augmentations.build_train_augmentation()` returns an Albumentations
pipeline with JPEG compression, Gaussian and motion blur, color jitter, noise,
and horizontal flips. Use it only for training; validation and inference should
use deterministic preprocessing.

## Validation

Run the unit tests (they create temporary synthetic data and do not require the
hackathon dataset):

```powershell
python -m unittest discover -s tests -v
```

## Handoff for Member 2 — model and training

1. Implement the spatial image encoder and a compact CNN for the FFT map, then
   concatenate their features and produce one fake probability per image.
2. Read `data/faces/manifest.csv`; resolve each `path` relative to `data/faces/`.
   Use `label` as the binary target and `fold` to select train/validation rows.
3. Apply `build_train_augmentation()` only to training images. Generate the
   frequency map from the same augmented image so both streams stay aligned.
4. Keep validation grouped by the saved fold; do not make a new frame-level
   split. Track AUC/F1 and report per-fold and mean results.
5. Add the training/evaluation/export entry points and tests without using
   external training data. Coordinate any changes to the manifest contract
   (`path`, `label`, `group_id`, `fold`) before changing it.
