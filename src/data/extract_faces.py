"""Extract and cache MTCNN face crops while preserving manifest metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.manifest import IMAGE_SUFFIXES, build_manifest


def select_face_box(
    boxes: np.ndarray | None,
    probabilities: np.ndarray | None,
    min_confidence: float,
) -> tuple[float, float, float, float] | None:
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
            and confidence >= min_confidence
            and np.all(np.isfinite(numeric_box))
            and numeric_box[2] > numeric_box[0]
            and numeric_box[3] > numeric_box[1]
        ):
            valid.append((numeric_box, confidence))
    if not valid:
        return None
    box, _ = max(
        valid,
        key=lambda item: (item[0][2] - item[0][0])
        * (item[0][3] - item[0][1]),
    )
    return tuple(float(value) for value in box)


def expand_box(
    box: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    margin: float,
) -> tuple[int, int, int, int]:
    """Expand an xyxy box by a fractional margin and clamp to image bounds."""
    if not 0 <= margin <= 1:
        raise ValueError("Margin must be between 0 and 1.")
    height, width = image_shape[:2]
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    left = max(0, int(np.floor(x1 - box_width * margin)))
    top = max(0, int(np.floor(y1 - box_height * margin)))
    right = min(width, int(np.ceil(x2 + box_width * margin)))
    bottom = min(height, int(np.ceil(y2 + box_height * margin)))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid face box after clipping: {box}")
    return left, top, right, bottom


def create_mtcnn(device: str) -> Any:
    try:
        import torch
        from facenet_pytorch import MTCNN
    except ImportError as error:
        raise RuntimeError(
            "MTCNN dependencies are missing; install them with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    if device == "auto":
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)
    return MTCNN(keep_all=True, device=selected_device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Raw image directory.")
    parser.add_argument("--out", type=Path, required=True, help="Face crop directory.")
    parser.add_argument("--metadata", type=Path, help="Optional manifest or folds CSV.")
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="Treat input as an unlabeled test image directory.",
    )
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device.")
    parser.add_argument(
        "--no-face",
        choices=("skip", "full-image"),
        default="skip",
        help="How to handle frames where MTCNN finds no confident face.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 0 <= args.margin <= 1:
        raise ValueError("--margin must be between 0 and 1.")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be between 0 and 1.")

    data_root = args.data.expanduser().resolve(strict=True)
    output_root = args.out.expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"Data path is not a directory: {data_root}")
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError("Output directory must not be inside the raw data directory.")

    if args.unlabeled:
        if args.metadata is not None:
            raise ValueError("--metadata cannot be combined with --unlabeled.")
        test_paths = sorted(
            path
            for path in data_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not test_paths:
            raise ValueError(f"No supported images found under {data_root}")
        manifest = pd.DataFrame(
            {"path": [path.relative_to(data_root).as_posix() for path in test_paths]}
        )
    else:
        manifest = build_manifest(
            data_root,
            metadata_path=args.metadata,
            path_column=args.path_column,
            label_column=args.label_column,
            group_column=args.group_column,
        )
    detector = create_mtcnn(args.device)
    output_root.mkdir(parents=True, exist_ok=True)

    extracted: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    fallbacks: list[dict[str, str]] = []
    written_paths: set[str] = set()
    for row in tqdm(manifest.to_dict("records"), desc="Extracting faces", unit="image"):
        source_relative = str(row["path"])
        source_path = data_root / Path(source_relative)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append({"path": source_relative, "reason": "unreadable image"})
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probabilities = detector.detect(rgb_image)
        face_box = select_face_box(boxes, probabilities, args.min_confidence)

        if face_box is None:
            if args.no_face == "skip":
                failures.append(
                    {"path": source_relative, "reason": "no confident face detected"}
                )
                continue
            fallbacks.append(
                {"path": source_relative, "reason": "no confident face; used full image"}
            )
            crop = image
        else:
            left, top, right, bottom = expand_box(face_box, image.shape, args.margin)
            crop = image[top:bottom, left:right]

        destination_relative = Path(source_relative)
        destination_key = destination_relative.as_posix().casefold()
        if destination_key in written_paths:
            failures.append(
                {"path": source_relative, "reason": "duplicate output crop path"}
            )
            continue
        destination_path = output_root / destination_relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination_path), crop):
            failures.append({"path": source_relative, "reason": "failed to write crop"})
            continue
        written_paths.add(destination_key)

        record = dict(row)
        if args.unlabeled:
            record["group_id"] = Path(source_relative).parent.as_posix()
            record["filename"] = source_relative
        record["original_path"] = source_relative
        record["path"] = destination_relative.as_posix()
        extracted.append(record)

    columns = list(manifest.columns)
    if args.unlabeled:
        columns = [column for column in columns if column != "label"]
        columns.extend(
            column for column in ("group_id", "filename") if column not in columns
        )
    if "original_path" not in columns:
        columns.append("original_path")
    base_columns = ["path", "group_id"] if args.unlabeled else ["path", "label", "group_id"]
    result_columns = base_columns + [
        column for column in columns if column not in base_columns
    ]
    pd.DataFrame(extracted, columns=result_columns).to_csv(
        output_root / "manifest.csv", index=False
    )
    pd.DataFrame(failures, columns=["path", "reason"]).to_csv(
        output_root / "failures.csv", index=False
    )
    pd.DataFrame(fallbacks, columns=["path", "reason"]).to_csv(
        output_root / "fallbacks.csv", index=False
    )

    print(
        f"Saved {len(extracted)} face crops to {output_root}; "
        f"{len(failures)} image(s) were skipped or failed, "
        f"{len(fallbacks)} used full-frame fallback."
    )
    if not extracted:
        raise RuntimeError(
            f"No face crops were created; inspect {output_root / 'failures.csv'}."
        )
    error_count = sum(
        failure["reason"] != "no confident face detected" for failure in failures
    )
    if error_count:
        raise RuntimeError(
            f"{error_count} extraction error(s); see {output_root / 'failures.csv'}."
        )


if __name__ == "__main__":
    main()
