# src/training/benchmark.py
import argparse
import gc
import json
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
import torch

from ..models.dual_stream_net import DualStreamNet


def _measure_pytorch(
    checkpoint_path: Path,
    image_size: int = 256,
    batch_size: int = 1,
    warmup_runs: int = 10,
    timed_runs: int = 50,
    device_name: str = "cpu",
) -> Dict[str, Any]:
    device = torch.device(device_name)
    gc.collect()
    tracemalloc.start()

    model = DualStreamNet(num_classes=1, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    rgb = torch.randn(batch_size, 3, image_size, image_size, device=device)
    freq = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(rgb, freq)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(rgb, freq)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    _, peak_ram_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mean_batch_ms = float(np.mean(latencies_ms))
    p95_batch_ms = float(np.percentile(latencies_ms, 95))
    per_image_ms = mean_batch_ms / batch_size
    throughput_fps = 1000.0 / per_image_ms
    disk_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    return {
        "runtime": f"PyTorch ({device.type.upper()})",
        "batch_size": batch_size,
        "parameters": total_params,
        "disk_size_mb": round(disk_mb, 2),
        "mean_latency_ms_per_image": round(per_image_ms, 2),
        "p95_batch_latency_ms": round(p95_batch_ms, 2),
        "throughput_img_per_sec": round(throughput_fps, 2),
        "peak_python_ram_mb": round(peak_ram_bytes / (1024 * 1024), 2),
    }


def _measure_onnx(
    onnx_path: Path,
    image_size: int = 256,
    batch_size: int = 1,
    warmup_runs: int = 10,
    timed_runs: int = 50,
) -> Dict[str, Any]:
    gc.collect()
    tracemalloc.start()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_options, providers=providers
    )

    rgb_np = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)
    freq_np = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)
    feed = {"rgb_image": rgb_np, "freq_image": freq_np}

    for _ in range(warmup_runs):
        _ = session.run(["logits"], feed)

    latencies_ms = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = session.run(["logits"], feed)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    _, peak_ram_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mean_batch_ms = float(np.mean(latencies_ms))
    p95_batch_ms = float(np.percentile(latencies_ms, 95))
    per_image_ms = mean_batch_ms / batch_size
    throughput_fps = 1000.0 / per_image_ms
    disk_mb = onnx_path.stat().st_size / (1024 * 1024)

    return {
        "runtime": f"ONNX Runtime ({session.get_providers()[0]})",
        "batch_size": batch_size,
        "parameters": None,
        "disk_size_mb": round(disk_mb, 2),
        "mean_latency_ms_per_image": round(per_image_ms, 2),
        "p95_batch_latency_ms": round(p95_batch_ms, 2),
        "throughput_img_per_sec": round(throughput_fps, 2),
        "peak_python_ram_mb": round(peak_ram_bytes / (1024 * 1024), 2),
    }


def run_benchmarks(
    checkpoint_path: str = "checkpoints/best_model_fold_0.pt",
    onnx_path: str = "onnx_models/best_model_fold_0.onnx",
    image_size: int = 256,
    warmup_runs: int = 3,
    timed_runs: int = 15,
    output_json: str = "benchmarks/efficiency_report.json",
) -> Dict[str, Any]:
    ckpt = Path(checkpoint_path)
    onnx_file = Path(onnx_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not onnx_file.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_file}")

    results = []
    for bs in (1, 8):
        print(f"⏱️  Benchmarking PyTorch (batch_size={bs}, runs={timed_runs})...")
        pt_res = _measure_pytorch(
            ckpt, image_size=image_size, batch_size=bs, warmup_runs=warmup_runs, timed_runs=timed_runs
        )
        print(f"⏱️  Benchmarking ONNX Runtime (batch_size={bs}, runs={timed_runs})...")
        ort_res = _measure_onnx(
            onnx_file, image_size=image_size, batch_size=bs, warmup_runs=warmup_runs, timed_runs=timed_runs
        )
        speedup = round(
            pt_res["mean_latency_ms_per_image"] / max(ort_res["mean_latency_ms_per_image"], 1e-6),
            2,
        )
        results.append(
            {
                "batch_size": bs,
                "pytorch": pt_res,
                "onnx_runtime": ort_res,
                "onnx_speedup_x": speedup,
            }
        )

    print("\n=== MODEL EFFICIENCY BENCHMARK (DualStreamNet 256x256) ===")
    for entry in results:
        bs = entry["batch_size"]
        pt = entry["pytorch"]
        ort_m = entry["onnx_runtime"]
        print(f"\n[Batch Size = {bs}]")
        print(
            f"  PyTorch CPU     : {pt['mean_latency_ms_per_image']:.2f} ms/img | "
            f"{pt['throughput_img_per_sec']:.1f} img/s | Size: {pt['disk_size_mb']} MB"
        )
        print(
            f"  ONNX Runtime    : {ort_m['mean_latency_ms_per_image']:.2f} ms/img | "
            f"{ort_m['throughput_img_per_sec']:.1f} img/s | Size: {ort_m['disk_size_mb']} MB | "
            f"Speedup: {entry['onnx_speedup_x']}x"
        )

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "DualStreamNet (EfficientNet-B2 + 2D-FFT Branch)",
        "input_resolution": f"3x{image_size}x{image_size}",
        "total_parameters": results[0]["pytorch"]["parameters"],
        "benchmarks": results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n📊 Benchmark report saved to {out_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime efficiency.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model_fold_0.pt")
    parser.add_argument("--onnx", default="onnx_models/best_model_fold_0.onnx")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output-json", default="benchmarks/efficiency_report.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmarks(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        image_size=args.image_size,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        output_json=args.output_json,
    )