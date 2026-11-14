# src/training/inference_pipeline.py
import inspect
import os
from typing import Iterable, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.dual_stream_net import DualStreamNet
from .run_training import HackathonDataset


def _discover_artifacts(
    artifact_dir: str,
    ext: str,
    folds: Optional[Iterable[int]] = None,
) -> list[str]:
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"Directory not found: {artifact_dir}")

    files = sorted(f for f in os.listdir(artifact_dir) if f.endswith(ext))
    if folds is not None:
        fold_tokens = {f"fold_{int(k)}" for k in folds}
        filtered = [f for f in files if any(token in f for token in fold_tokens)]
        if filtered:
            files = filtered

    paths = [os.path.join(artifact_dir, f) for f in files]
    if not paths:
        raise FileNotFoundError(f"No {ext} files found in {artifact_dir}.")
    return paths


def _normalize_manifest_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            sample = str(non_null.iloc[0]).replace("\\", "/")
            if "/data/" in sample:
                df[col] = df[col].apply(
                    lambda p: p
                    if os.path.exists(str(p))
                    else (
                        "data/" + str(p).replace("\\", "/").split("/data/", 1)[1]
                        if "/data/" in str(p).replace("\\", "/")
                        else p
                    )
                )
    return df


def _build_test_dataset(test_df: pd.DataFrame) -> HackathonDataset:
    test_df = _normalize_manifest_paths(test_df)
    sig = inspect.signature(HackathonDataset.__init__)
    kwargs = {}
    if "is_train" in sig.parameters:
        kwargs["is_train"] = False
    elif "train" in sig.parameters:
        kwargs["train"] = False
    if "transform" in sig.parameters:
        kwargs["transform"] = None
    return HackathonDataset(test_df, **kwargs)


def _extract_image_ids(test_df: pd.DataFrame) -> list[str]:
    for col in ("image_id", "filename", "id", "filepath", "path"):
        if col in test_df.columns:
            return [os.path.basename(str(val)) for val in test_df[col].tolist()]
    raise KeyError(f"Could not find ID column in test manifest: {list(test_df.columns)}")


def run_ensemble_inference(
    test_manifest_csv: str = "test_manifest.csv",
    checkpoint_dir: str = "checkpoints",
    output_csv: str = "submission.csv",
    batch_size: int = 32,
    use_tta: bool = True,
    folds: Optional[Iterable[int]] = None,
    num_workers: int = 2,
    use_onnx: bool = False,
    onnx_dir: str = "onnx_models",
) -> pd.DataFrame:
    if not os.path.isfile(test_manifest_csv):
        raise FileNotFoundError(f"Test manifest CSV not found: {test_manifest_csv}")

    test_df = pd.read_csv(test_manifest_csv)
    image_ids = _extract_image_ids(test_df)
    test_dataset = _build_test_dataset(test_df)

    results = []
    processed = 0

    if use_onnx:
        onnx_paths = _discover_artifacts(onnx_dir, ".onnx", folds=folds)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        sessions = [
            ort.InferenceSession(p, sess_options=sess_opts, providers=providers)
            for p in onnx_paths
        ]
        print(f"🚀 Running ONNX Ensemble ({len(sessions)} models, TTA={use_tta}) on {providers[0]}")

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in tqdm(test_loader, desc="ONNX Ensemble Inference"):
            rgb_np = batch[0].numpy()
            freq_np = batch[1].numpy()
            bs = rgb_np.shape[0]

            if use_tta:
                rgb_flip = np.ascontiguousarray(rgb_np[:, :, :, ::-1])
                freq_flip = np.ascontiguousarray(freq_np[:, :, :, ::-1])

            batch_preds = np.zeros(bs, dtype=np.float32)
            for sess in sessions:
                logits = sess.run(["logits"], {"rgb_image": rgb_np, "freq_image": freq_np})[0].reshape(-1)
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
                if use_tta:
                    logits_f = sess.run(["logits"], {"rgb_image": rgb_flip, "freq_image": freq_flip})[0].reshape(-1)
                    probs_f = 1.0 / (1.0 + np.exp(-np.clip(logits_f, -60.0, 60.0)))
                    probs = 0.5 * (probs + probs_f)
                batch_preds += probs

            batch_preds = np.clip(np.nan_to_num(batch_preds / len(sessions), nan=0.5), 0.0, 1.0)
            for img_id, prob in zip(image_ids[processed : processed + bs], batch_preds):
                results.append({"image_id": img_id, "label": float(prob)})
            processed += bs
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔍 Running PyTorch Ensemble on {device} | TTA={use_tta}")
        checkpoint_paths = _discover_artifacts(checkpoint_dir, ".pt", folds=folds)

        models = []
        for path in checkpoint_paths:
            model = DualStreamNet(num_classes=1, pretrained=False).to(device)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers if device.type == "cuda" else 0,
            pin_memory=device.type == "cuda",
        )

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Ensemble Inference"):
                rgb = batch[0].to(device, non_blocking=True)
                freq = batch[1].to(device, non_blocking=True)
                if use_tta:
                    rgb_flip = torch.flip(rgb, dims=[-1])
                    freq_flip = torch.flip(freq, dims=[-1])

                batch_preds = torch.zeros(rgb.size(0), device=device, dtype=torch.float32)
                for model in models:
                    with autocast(device_type=device.type, enabled=device.type == "cuda"):
                        probs = torch.sigmoid(model(rgb, freq)).flatten().float()
                        if use_tta:
                            probs_flip = torch.sigmoid(model(rgb_flip, freq_flip)).flatten().float()
                            probs = 0.5 * (probs + probs_flip)
                    batch_preds += probs

                batch_preds = torch.clamp(torch.nan_to_num(batch_preds / len(models), nan=0.5), 0.0, 1.0)
                for img_id, prob in zip(image_ids[processed : processed + rgb.size(0)], batch_preds.cpu().numpy()):
                    results.append({"image_id": img_id, "label": float(prob)})
                processed += rgb.size(0)

    if processed != len(image_ids):
        raise RuntimeError(f"Inference produced {processed} predictions for {len(image_ids)} test rows.")

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Final submission saved to {output_csv} ({len(out_df)} rows)")
    return out_df