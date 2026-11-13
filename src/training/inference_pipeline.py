# src/training/inference_pipeline.py
import os
import torch
import pandas as pd
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.dual_stream_net import DualStreamNet
from .run_training import HackathonDataset

def run_ensemble_inference(test_manifest_csv="test_manifest.csv", checkpoint_dir="checkpoints", output_csv="submission.csv", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Running Ensemble Inference on device: {device}")

    checkpoint_paths = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    
    if not checkpoint_paths:
        print("⚠️ No checkpoints found! Please run training first.")
        return

    models = []
    for path in checkpoint_paths:
        print(f"Loading checkpoint: {path}")
        model = DualStreamNet(num_classes=1, pretrained=False).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} fold models for ensemble voting.")
    
    test_df = pd.read_csv(test_manifest_csv)
    test_dataset = HackathonDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    results = []
    filenames = test_df["filename"].tolist()
    processed = 0
    with torch.no_grad():
        for rgb, freq, _ in tqdm(test_loader, desc="Ensemble Inference"):
            rgb = rgb.to(device, non_blocking=True)
            freq = freq.to(device, non_blocking=True)
            
            batch_preds = torch.zeros(rgb.size(0), device=device)
            for model in models:
                with autocast():
                    outputs = model(rgb, freq)
                    probs = torch.sigmoid(outputs).flatten()
                    batch_preds += probs
                    
            batch_preds /= len(models)
            
            batch_filenames = filenames[processed : processed + rgb.size(0)]
            if len(batch_filenames) != rgb.size(0):
                raise RuntimeError("Test manifest filenames do not match inference rows.")
            for fname, prob in zip(batch_filenames, batch_preds.cpu().numpy()):
                results.append({"filename": fname, "label": float(prob)})
            processed += rgb.size(0)

    if processed != len(filenames):
        raise RuntimeError(
            f"Inference produced {processed} predictions for {len(filenames)} test rows."
        )

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Final submission saved successfully to {output_csv}!")

if __name__ == "__main__":
    run_ensemble_inference()