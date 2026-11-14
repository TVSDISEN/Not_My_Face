# src/training/export_onnx.py
import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ..models.dual_stream_net import DualStreamNet


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    image_size: int = 256,
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    ckpt_path = Path(checkpoint_path)
    out_path = Path(onnx_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    model = DualStreamNet(num_classes=1, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()

    dummy_rgb = torch.randn(1, 3, image_size, image_size, dtype=torch.float32, device=device)
    dummy_freq = torch.randn(1, 3, image_size, image_size, dtype=torch.float32, device=device)

    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["rgb_image", "freq_image"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "rgb_image": {0: "batch_size"},
            "freq_image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    }

    try:
        torch.onnx.export(
            model,
            (dummy_rgb, dummy_freq),
            str(out_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            model,
            (dummy_rgb, dummy_freq),
            str(out_path),
            **export_kwargs,
        )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✅ Exported {ckpt_path.name} -> {out_path} ({size_mb:.2f} MB)")

    if verify:
        verify_onnx_parity(model, out_path, image_size=image_size)

    return out_path


def verify_onnx_parity(
    pt_model: torch.nn.Module,
    onnx_path: str | Path,
    image_size: int = 256,
    prob_atol: float = 1e-4,
) -> float:
    """Verify numerical parity between PyTorch and ONNX Runtime across batch sizes."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_prob_diff = 0.0

    for batch_size in (1, 4):
        # Realistic [0, 1] scaled inputs rather than extreme out-of-distribution noise
        rgb = torch.rand(batch_size, 3, image_size, image_size, dtype=torch.float32)
        freq = torch.rand(batch_size, 3, image_size, image_size, dtype=torch.float32)

        with torch.inference_mode():
            pt_logits = pt_model(rgb, freq).numpy()

        ort_logits = session.run(
            ["logits"],
            {"rgb_image": rgb.numpy(), "freq_image": freq.numpy()},
        )[0]

        pt_probs = 1.0 / (1.0 + np.exp(-np.clip(pt_logits, -60.0, 60.0)))
        ort_probs = 1.0 / (1.0 + np.exp(-np.clip(ort_logits, -60.0, 60.0)))

        prob_diff = float(np.max(np.abs(pt_probs - ort_probs)))
        max_prob_diff = max(max_prob_diff, prob_diff)

        np.testing.assert_allclose(pt_logits, ort_logits, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(pt_probs, ort_probs, atol=prob_atol)

    print(f"🔍 Parity verified (max probability diff = {max_prob_diff:.2e} <= {prob_atol:.0e})")
    return max_prob_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DualStreamNet checkpoints to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Single .pt checkpoint to export. If omitted, exports all folds in --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing .pt checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_models",
        help="Destination folder for exported .onnx models.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        out_file = Path(args.output_dir) / f"{ckpt.stem}.onnx"
        export_checkpoint_to_onnx(
            ckpt,
            out_file,
            image_size=args.image_size,
            opset_version=args.opset,
            verify=not args.skip_verify,
        )
        return

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No .pt files found in {ckpt_dir}")

    for ckpt in ckpts:
        out_file = Path(args.output_dir) / f"{ckpt.stem}.onnx"
        export_checkpoint_to_onnx(
            ckpt,
            out_file,
            image_size=args.image_size,
            opset_version=args.opset,
            verify=not args.skip_verify,
        )


if __name__ == "__main__":
    main()