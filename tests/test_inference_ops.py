# tests/test_inference_ops.py
import os
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models.dual_stream_net import DualStreamNet
from src.training.export_onnx import export_checkpoint_to_onnx
from src.training.inference_pipeline import (
    _normalize_manifest_paths,
    run_ensemble_inference,
)
from src.training.predict import _align_to_sample_submission


def _create_dummy_checkpoint(ckpt_path: Path) -> None:
    model = DualStreamNet(num_classes=1, pretrained=False)
    torch.save(
        {
            "fold": 0,
            "epoch": 1,
            "val_loss": 0.5,
            "val_acc": 0.8,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )


def _create_dummy_image(img_path: Path, size: int = 256) -> None:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)


def test_onnx_export_and_parity():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ckpt_path = tmp / "best_model_fold_0.pt"
        onnx_path = tmp / "best_model_fold_0.onnx"

        _create_dummy_checkpoint(ckpt_path)
        exported = export_checkpoint_to_onnx(
            ckpt_path, onnx_path, image_size=256, verify=True
        )
        assert exported.is_file()
        assert exported.stat().st_size > 1_000_000

        session = ort.InferenceSession(
            str(exported), providers=["CPUExecutionProvider"]
        )
        rgb = np.random.rand(2, 3, 256, 256).astype(np.float32)
        freq = np.random.rand(2, 3, 256, 256).astype(np.float32)
        out = session.run(["logits"], {"rgb_image": rgb, "freq_image": freq})[0]
        assert out.shape == (2, 1)
        assert not np.isnan(out).any()


def test_colab_path_normalization():
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            local_img = Path("data/faces/test/sample.jpg")
            local_img.parent.mkdir(parents=True, exist_ok=True)
            _create_dummy_image(local_img)

            df = pd.DataFrame(
                {
                    "image_id": ["sample.jpg"],
                    "rgb_path": ["/content/Not_My_Face/data/faces/test/sample.jpg"],
                    "freq_path": ["/content/Not_My_Face/data/faces/test/sample.jpg"],
                }
            )
            normalized = _normalize_manifest_paths(df)
            assert normalized["rgb_path"].iloc[0] == "data/faces/test/sample.jpg"
        finally:
            os.chdir(old_cwd)


def test_end_to_end_ensemble_inference_and_alignment():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ckpt_dir = tmp / "checkpoints"
        ckpt_dir.mkdir()
        _create_dummy_checkpoint(ckpt_dir / "best_model_fold_0.pt")
        _create_dummy_checkpoint(ckpt_dir / "best_model_fold_1.pt")

        img_dir = tmp / "images"
        img_dir.mkdir()
        rows = []
        for idx in range(4):
            fname = f"test_{idx:02d}.jpg"
            rgb_p = img_dir / f"rgb_{fname}"
            freq_p = img_dir / f"freq_{fname}"
            _create_dummy_image(rgb_p)
            _create_dummy_image(freq_p)
            rows.append(
                {
                    "image_id": fname,
                    "rgb_path": str(rgb_p),
                    "freq_path": str(freq_p),
                    "label": 0,
                }
            )

        manifest_csv = tmp / "test_manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_csv, index=False)

        out_csv = tmp / "submission.csv"
        pred_df = run_ensemble_inference(
            test_manifest_csv=str(manifest_csv),
            checkpoint_dir=str(ckpt_dir),
            output_csv=str(out_csv),
            batch_size=2,
            use_tta=True,
            num_workers=0,
        )

        assert out_csv.is_file()
        assert len(pred_df) == 4
        assert list(pred_df.columns) == ["image_id", "label"]
        assert not pred_df["label"].isna().any()
        assert ((pred_df["label"] >= 0.0) & (pred_df["label"] <= 1.0)).all()

        # Verify alignment to a sample_submission with custom column names & order
        sample_csv = tmp / "sample_submission.csv"
        pd.DataFrame(
            {"filename": ["test_03.jpg", "test_00.jpg"], "prediction": [0.0, 0.0]}
        ).to_csv(sample_csv, index=False)

        aligned_csv = tmp / "aligned_submission.csv"
        aligned_df = _align_to_sample_submission(
            pred_df, str(sample_csv), None, str(aligned_csv)
        )
        assert list(aligned_df.columns) == ["filename", "prediction"]
        assert aligned_df["filename"].tolist() == ["test_03.jpg", "test_00.jpg"]