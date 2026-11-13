"""Generate Member 2's training and test manifests plus cached FFT images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.frequency import compute_fft_spectrogram, frequency_map_to_rgb
from src.data.manifest import IMAGE_SUFFIXES


def _resolve_crop(value: object, root: Path) -> tuple[Path, str]:
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Crop paths cannot be empty.")
    candidate = Path(str(value).strip())
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"Face crop does not exist: {candidate}") from error
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Face crop must be inside {root}: {resolved}") from error
    if not resolved.is_file() or resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Not a supported image file: {resolved}")
    return resolved, relative.as_posix()


def _save_frequency_image(rgb_path: Path, relative_path: str, frequency_root: Path) -> Path:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read face crop: {rgb_path}")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frequency_map = compute_fft_spectrogram(rgb_image)
    frequency_rgb = frequency_map_to_rgb(frequency_map)

    relative = Path(relative_path)
    frequency_relative = relative.with_name(
        f"{relative.stem}{relative.suffix.lower()}.png"
    )
    frequency_path = frequency_root / frequency_relative
    frequency_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frequency_rgb).save(frequency_path, format="PNG")
    return frequency_path.resolve()


def _load_face_rows(faces_root: Path, is_test: bool) -> pd.DataFrame:
    manifest_path = faces_root / "manifest.csv"
    if not manifest_path.is_file():
        raise ValueError(f"Face manifest not found: {manifest_path}")
    manifest = pd.read_csv(
        manifest_path,
        dtype={"path": "string", "group_id": "string", "filename": "string"},
    )
    if manifest.empty:
        raise ValueError(f"Face manifest contains no rows: {manifest_path}")
    if "path" not in manifest:
        raise ValueError(f"{manifest_path} must contain a 'path' column.")

    if not is_test:
        required = {"label", "group_id", "fold"}
        missing = required.difference(manifest.columns)
        if missing:
            raise ValueError(
                f"Training face manifest is missing: {', '.join(sorted(missing))}"
            )
        labels = pd.to_numeric(manifest["label"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("Training labels must be binary 0 (real) or 1 (fake).")
        if labels.nunique() != 2:
            raise ValueError("Training manifest must contain both real and fake samples.")
        folds = pd.to_numeric(manifest["fold"], errors="coerce")
        if folds.isna().any() or not folds.isin(range(5)).all():
            raise ValueError("Training fold values must be integers from 0 to 4.")
        if set(folds.astype(int).unique()) != set(range(5)):
            raise ValueError("Training manifest must contain samples in all five folds.")
        if manifest["group_id"].isna().any() or manifest["group_id"].str.strip().eq("").any():
            raise ValueError("Training group IDs must not be empty.")
        groups_per_fold = manifest.assign(fold=folds.astype(int)).groupby("group_id")[
            "fold"
        ].nunique()
        if (groups_per_fold != 1).any():
            raise ValueError("A training group appears in more than one fold.")
        manifest["label"] = labels.astype(int)
        manifest["fold"] = folds.astype(int)

    return manifest


def prepare_manifests(
    train_faces: str | Path,
    test_faces: str | Path,
    output_dir: str | Path,
    frequency_dir: str | Path,
) -> tuple[Path, Path]:
    """Validate prepared crops and produce absolute-path manifests and FFT maps."""
    train_root = Path(train_faces).expanduser().resolve(strict=True)
    test_root = Path(test_faces).expanduser().resolve(strict=True)
    output_root = Path(output_dir).expanduser().resolve()
    frequency_root = Path(frequency_dir).expanduser().resolve()
    for root in (train_root, test_root):
        if not root.is_dir():
            raise ValueError(f"Face crop path is not a directory: {root}")
    if any(
        root == source or root in source.parents
        for root in (output_root, frequency_root)
        for source in (train_root, test_root)
    ):
        raise ValueError("Output and frequency directories must not be face input roots.")
    if (
        output_root == frequency_root
        or output_root in frequency_root.parents
        or frequency_root in output_root.parents
    ):
        raise ValueError("Manifest and frequency output directories must be separate.")

    train_manifest = _load_face_rows(train_root, is_test=False)
    test_manifest = _load_face_rows(test_root, is_test=True)
    output_root.mkdir(parents=True, exist_ok=True)
    frequency_root.mkdir(parents=True, exist_ok=True)

    train_records = []
    seen_train_paths: set[str] = set()
    for row in tqdm(
        train_manifest.to_dict("records"),
        desc="Preparing training frequency maps",
        unit="image",
    ):
        rgb_path, relative_path = _resolve_crop(row["path"], train_root)
        if relative_path.casefold() in seen_train_paths:
            raise ValueError(f"Duplicate training crop path: {relative_path}")
        seen_train_paths.add(relative_path.casefold())
        freq_path = _save_frequency_image(
            rgb_path, f"train/{relative_path}", frequency_root
        )
        train_records.append(
            {
                "rgb_path": str(rgb_path),
                "freq_path": str(freq_path),
                "fold": int(row["fold"]),
                "label": int(row["label"]),
            }
        )

    test_records = []
    seen_test_paths: set[str] = set()
    for row in tqdm(
        test_manifest.to_dict("records"),
        desc="Preparing test frequency maps",
        unit="image",
    ):
        rgb_path, relative_path = _resolve_crop(row["path"], test_root)
        if relative_path.casefold() in seen_test_paths:
            raise ValueError(f"Duplicate test crop path: {relative_path}")
        seen_test_paths.add(relative_path.casefold())
        freq_path = _save_frequency_image(
            rgb_path, f"test/{relative_path}", frequency_root
        )
        filename = row.get("filename")
        if pd.isna(filename) or not str(filename).strip():
            filename = row.get("original_path")
        if pd.isna(filename) or not str(filename).strip():
            filename = relative_path
        test_records.append(
            {
                "rgb_path": str(rgb_path),
                "freq_path": str(freq_path),
                "filename": str(filename),
            }
        )

    filenames = pd.Series([record["filename"] for record in test_records])
    if filenames.duplicated().any():
        duplicates = filenames[filenames.duplicated(keep=False)].unique()
        raise ValueError(
            "Test filenames must be unique for inference; duplicates include "
            + ", ".join(str(value) for value in duplicates[:5])
        )

    train_output = output_root / "manifest_folds.csv"
    test_output = output_root / "test_manifest.csv"
    pd.DataFrame(
        train_records, columns=["rgb_path", "freq_path", "fold", "label"]
    ).to_csv(train_output, index=False)
    pd.DataFrame(
        test_records, columns=["rgb_path", "freq_path", "filename"]
    ).to_csv(test_output, index=False)
    print(
        f"Wrote {len(train_records)} training rows to {train_output} and "
        f"{len(test_records)} test rows to {test_output}."
    )
    print(f"Cached frequency maps under {frequency_root}.")
    return train_output, test_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-faces", type=Path, required=True)
    parser.add_argument("--test-faces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frequency-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_manifests(
        args.train_faces,
        args.test_faces,
        args.out_dir,
        args.frequency_dir,
    )


if __name__ == "__main__":
    main()
