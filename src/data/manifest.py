"""Shared image-manifest parsing and validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_LABELS = {"0", "real", "authentic", "original"}
FAKE_LABELS = {"1", "fake", "deepfake", "synthetic"}


def normalize_label(value: Any) -> int:
    """Convert accepted real/fake labels to 0/1, rejecting unknown values."""
    if pd.isna(value):
        raise ValueError("Labels cannot be empty.")

    token = str(value).strip().lower()
    if token in REAL_LABELS:
        return 0
    if token in FAKE_LABELS:
        return 1
    try:
        numeric_label = float(token)
    except ValueError:
        numeric_label = -1
    if numeric_label == 0:
        return 0
    if numeric_label == 1:
        return 1
    raise ValueError(
        f"Unsupported label {value!r}; use real/fake or 0/1 "
        "(0 = real, 1 = fake)."
    )


def _resolve_image_path(value: Any, root: Path) -> tuple[Path, str]:
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Manifest image paths cannot be empty.")

    candidate = Path(str(value).strip())
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"Image does not exist: {candidate}") from error

    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Image path must be inside the data directory: {resolved}"
        ) from error

    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image extension: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Image path is not a file: {resolved}")

    return resolved, relative.as_posix()


def _class_component(parts: tuple[str, ...], image_path: Path) -> tuple[int, int]:
    matches = [
        (index, normalize_label(part))
        for index, part in enumerate(parts)
        if part.strip().lower() in REAL_LABELS | FAKE_LABELS
    ]
    labels = {label for _, label in matches}
    if not matches:
        raise ValueError(
            f"Cannot infer real/fake label from {image_path}; "
            "use class directories or provide --metadata."
        )
    if len(labels) != 1:
        raise ValueError(
            f"Conflicting class directories in {image_path}; provide --metadata."
        )
    return matches[0]


def _infer_group(parts: tuple[str, ...], class_index: int, image_path: Path) -> str:
    if class_index + 1 < len(parts):
        return parts[class_index + 1]

    # Flat class folders often encode frame numbers in filenames. Strip a
    # conventional frame suffix when possible; explicit metadata is safer.
    stem = re.sub(
        r"(?i)(?:[_-]?(?:frame|img|image|f)?[_-]?\d+)$",
        "",
        image_path.stem,
    )
    return stem or image_path.stem


def build_manifest(
    data_dir: str | Path,
    metadata_path: str | Path | None = None,
    path_column: str = "path",
    label_column: str = "label",
    group_column: str = "group_id",
) -> pd.DataFrame:
    """Return a validated manifest with relative paths and binary labels.

    Without metadata, labels are read from real/fake class directories and the
    first directory below the class directory is used as the group ID.
    """
    root = Path(data_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Data path is not a directory: {root}")

    if metadata_path is not None:
        metadata_file = Path(metadata_path).expanduser().resolve(strict=True)
        source = pd.read_csv(
            metadata_file,
            dtype={path_column: "string", group_column: "string"},
        )
        required = {path_column, label_column, group_column}
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(
                f"Metadata is missing required column(s): {', '.join(sorted(missing))}"
            )
        if source.empty:
            raise ValueError(f"Metadata contains no rows: {metadata_file}")

        rows: list[dict[str, Any]] = []
        passthrough = [
            column
            for column in source.columns
            if column not in {path_column, label_column, group_column}
            and column not in {"path", "label", "group_id"}
        ]
        seen_paths: set[str] = set()
        for row_number, row in source.iterrows():
            resolved, relative = _resolve_image_path(row[path_column], root)
            if relative in seen_paths:
                raise ValueError(f"Duplicate image path in metadata: {relative}")
            seen_paths.add(relative)
            group_id = row[group_column]
            if pd.isna(group_id) or not str(group_id).strip():
                raise ValueError(f"Empty group ID on metadata row {row_number + 2}.")
            record: dict[str, Any] = {
                "path": relative,
                "label": normalize_label(row[label_column]),
                "group_id": str(group_id).strip(),
            }
            record.update({column: row[column] for column in passthrough})
            rows.append(record)
        return pd.DataFrame(rows)

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No supported images found under {root}")

    rows = []
    for image_path in image_paths:
        relative = image_path.relative_to(root)
        parts = relative.parts[:-1]
        class_index, label = _class_component(parts, image_path)
        rows.append(
            {
                "path": relative.as_posix(),
                "label": label,
                "group_id": _infer_group(parts, class_index, image_path),
            }
        )
    return pd.DataFrame(rows)
