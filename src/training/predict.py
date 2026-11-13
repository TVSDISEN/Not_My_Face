# src/predict.py
import os
import torch
import pandas as pd
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dual_stream_net import DualStreamNet

def generate_predictions(model_path, test_dataloader, output_csv="submission.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model weights from {model_path}...")
    
    model = DualStreamNet(num_classes=1, pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict robustly
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    results = []
    with torch.no_grad():
        for rgb, freq, file_names in tqdm(test_dataloader, desc="Generating Predictions"):
            rgb = rgb.to(device, non_blocking=True)
            freq = freq.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(rgb, freq)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                
            for fname, prob in zip(file_names, probs):
                results.append({"filename": fname, "label": prob})
                
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions successfully saved to {output_csv}!")

if __name__ == "__main__":
    print("Inference script template ready.")