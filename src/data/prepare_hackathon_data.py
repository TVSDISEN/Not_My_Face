"""Validate the supplied hackathon dataset and create fold-input metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.manifest import IMAGE_SUFFIXES, normalize_label


def _image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Expected image directory does not exist: {directory}")
    images = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    result: dict[str, Path] = {}
    for path in images:
        image_id = path.stem
        if image_id in result:
            raise ValueError(f"Duplicate image ID in {directory}: {image_id}")
        result[image_id] = path
    return result


def prepare_hackathon_data(data_root: str | Path, output_dir: str | Path) -> Path:
    """Validate metadata/images and create path,label,group_id training CSV."""
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")
    train_root = root / "train"
    train_metadata_path = root / "train_metadata.csv"
    sample_path = root / "sample_submission.csv"
    if not train_metadata_path.is_file():
        raise ValueError(f"Training metadata not found: {train_metadata_path}")
    if not sample_path.is_file():
        raise ValueError(f"Sample submission not found: {sample_path}")

    metadata = pd.read_csv(
        train_metadata_path,
        dtype={"image_id": "string", "label": "string"},
    )
    if metadata.empty or not {"image_id", "label"}.issubset(metadata.columns):
        raise ValueError("train_metadata.csv must have non-empty image_id and label columns.")
    if metadata["image_id"].isna().any() or metadata["image_id"].str.strip().eq("").any():
        raise ValueError("Training metadata contains an empty image ID.")
    if metadata["image_id"].duplicated().any():
        raise ValueError("Training metadata contains duplicate image IDs.")

    metadata["image_id"] = metadata["image_id"].str.strip()
    metadata["label"] = metadata["label"].map(normalize_label)
    class_directories = {0: "real", 1: "fake"}
    class_images = {
        label: _image_files(train_root / class_name)
        for label, class_name in class_directories.items()
    }
    duplicate_class_ids = set(class_images[0]).intersection(class_images[1])
    if duplicate_class_ids:
        raise ValueError(
            "Image IDs occur in both real and fake directories: "
            + ", ".join(sorted(duplicate_class_ids)[:5])
        )

    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in metadata.to_dict("records"):
        image_id = str(row["image_id"])
        label = int(row["label"])
        class_name = class_directories[label]
        image_path = class_images[label].get(image_id)
        if image_path is None:
            other_class = class_directories[1 - label]
            if image_id in class_images[1 - label]:
                raise ValueError(
                    f"Image {image_id} is in {other_class}, but metadata label is {label}."
                )
            raise ValueError(
                f"Training image {image_id} is missing from train/{class_name}."
            )
        seen_ids.add(image_id)
        records.append(
            {
                "path": image_path.relative_to(train_root).as_posix(),
                "label": label,
                "group_id": image_id,
            }
        )

    actual_train_ids = set().union(*(set(images) for images in class_images.values()))
    if actual_train_ids != seen_ids:
        extra_ids = sorted(actual_train_ids - seen_ids)
        raise ValueError(
            f"Found {len(extra_ids)} training image(s) absent from metadata; "
            f"examples: {', '.join(extra_ids[:5])}."
        )

    sample = pd.read_csv(sample_path, dtype={"image_id": "string"})
    if sample.empty or "image_id" not in sample:
        raise ValueError("sample_submission.csv must contain a non-empty image_id column.")
    if sample["image_id"].isna().any() or sample["image_id"].str.strip().eq("").any():
        raise ValueError("Sample submission contains an empty image ID.")
    if sample["image_id"].duplicated().any():
        raise ValueError("Sample submission contains duplicate image IDs.")
    sample_ids = set(sample["image_id"].str.strip())
    test_images = _image_files(root / "test" / "images")
    test_ids = set(test_images)
    if sample_ids != test_ids:
        missing = sorted(sample_ids - test_ids)
        extra = sorted(test_ids - sample_ids)
        raise ValueError(
            "Test images do not match sample_submission.csv: "
            f"{len(missing)} missing, {len(extra)} extra."
        )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "train_metadata.csv"
    pd.DataFrame(records, columns=["path", "label", "group_id"]).to_csv(
        output_path, index=False
    )
    print(
        f"Validated {len(records)} training images "
        f"({sum(record['label'] == 0 for record in records)} real, "
        f"{sum(record['label'] == 1 for record in records)} fake) and "
        f"{len(test_ids)} test images."
    )
    print(
        "No video/subject identifiers are provided; each unique image_id is "
        "assigned as its own group."
    )
    print(f"Wrote fold-input metadata to {output_path}.")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_hackathon_data(args.data_root, args.out_dir)


if __name__ == "__main__":
    main()
