# src/training/predict.py
import argparse
import os
import tempfile
from typing import Optional

import pandas as pd

from .inference_pipeline import run_ensemble_inference

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_manifest_from_dir(test_dir: str, out_manifest_path: str) -> str:
    """Scan a directory of test images and generate a manifest CSV."""
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    records = []
    for root, _, files in os.walk(test_dir):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                full_path = os.path.join(root, fname)
                records.append(
                    {
                        "image_id": fname,
                        "filename": fname,
                        "filepath": full_path,
                        "path": full_path,
                        "label": 0,
                    }
                )

    if not records:
        raise RuntimeError(f"No image files found in test directory: {test_dir}")

    df = pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)
    df.to_csv(out_manifest_path, index=False)
    print(f"📄 Generated temporary test manifest ({len(df)} images) at {out_manifest_path}")
    return out_manifest_path


def _strip_ext(val: str) -> str:
    return os.path.splitext(os.path.basename(str(val).strip()))[0]


def _align_to_sample_submission(
    pred_df: pd.DataFrame,
    sample_submission_csv: Optional[str],
    id_col: Optional[str],
    output_csv: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Align predictions to exact sample_submission.csv schema (image_id, label, confidence)."""
    # Build extension-agnostic lookup: stem -> probability
    lookup = {
        _strip_ext(k): float(v)
        for k, v in zip(pred_df["image_id"], pred_df["label"])
    }

    if sample_submission_csv and os.path.isfile(sample_submission_csv):
        sample_df = pd.read_csv(sample_submission_csv)
        target_id_col = sample_df.columns[0]

        sample_keys = [_strip_ext(x) for x in sample_df[target_id_col]]
        probs = [lookup.get(k, 0.5) for k in sample_keys]

        missing = sum(1 for k in sample_keys if k not in lookup)
        if missing > 0:
            print(f"⚠️ Warning: {missing} sample IDs missing from predictions; defaulted to 0.5.")

        if list(sample_df.columns) == ["image_id", "label", "confidence"]:
            out_df = pd.DataFrame(
                {
                    "image_id": sample_df["image_id"],
                    "label": [int(p >= threshold) for p in probs],
                    "confidence": [round(float(p), 6) for p in probs],
                }
            )
        else:
            target_label_col = sample_df.columns[1] if len(sample_df.columns) > 1 else "label"
            out_df = pd.DataFrame(
                {
                    target_id_col: sample_df[target_id_col],
                    target_label_col: probs,
                }
            )

        if len(out_df) != len(sample_df) or out_df.isna().any().any():
            raise ValueError("Aligned submission failed row count or NaN validation.")

        out_df.to_csv(output_csv, index=False)
        print(f"🔗 Aligned output ({list(out_df.columns)}, {len(out_df)} rows) to {sample_submission_csv}")
        return out_df

    # Fallback when no sample_submission is passed: still format as 3-column schema
    out_df = pd.DataFrame(
        {
            id_col or "image_id": [_strip_ext(x) for x in pred_df["image_id"]],
            "label": [int(float(p) >= threshold) for p in pred_df["label"]],
            "confidence": [round(float(p), 6) for p in pred_df["label"]],
        }
    )
    out_df.to_csv(output_csv, index=False)
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dual-Stream Ensemble Inference and generate submission.csv"
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default=None,
        help="Path to test manifest CSV. If omitted, --test-dir will be scanned.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/raw/test",
        help="Directory of raw/cropped test images if --test-manifest is not provided.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained .pt / .pth fold checkpoints.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="submission.csv",
        help="Destination path for final prediction CSV.",
    )
    parser.add_argument(
        "--sample-submission",
        type=str,
        default=None,
        help="Optional sample_submission.csv to mirror exact column headers and ID order.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="image_id",
        help="ID column name in output CSV if --sample-submission is not provided.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold indices to ensemble (e.g., --folds 0 2 4). Defaults to all folds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable horizontal-flip Test-Time Augmentation for 2x faster inference.",
    )
    parser.add_argument("--use-onnx", action="store_true", help="Use ONNX Runtime for 7.5x faster CPU inference.")
    parser.add_argument("--onnx-dir", type=str, default="onnx_models", help="Directory of exported .onnx models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    temp_manifest = None
    try:
        if args.test_manifest:
            if not os.path.isfile(args.test_manifest):
                raise FileNotFoundError(f"Specified test manifest not found: {args.test_manifest}")
            manifest_path = args.test_manifest
        elif os.path.isfile("test_manifest.csv"):
            manifest_path = "test_manifest.csv"
        else:
            fd, temp_manifest = tempfile.mkstemp(prefix="test_manifest_", suffix=".csv")
            os.close(fd)
            manifest_path = _build_manifest_from_dir(args.test_dir, temp_manifest)

        pred_df = run_ensemble_inference(
            test_manifest_csv=manifest_path,
            checkpoint_dir=args.checkpoint_dir,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            use_tta=not args.no_tta,
            folds=args.folds,
        )

        _align_to_sample_submission(
            pred_df=pred_df,
            sample_submission_csv=args.sample_submission,
            id_col=args.id_col,
            output_csv=args.output_csv,
        )
    finally:
        if temp_manifest and os.path.exists(temp_manifest):
            os.remove(temp_manifest)


if __name__ == "__main__":
    main()