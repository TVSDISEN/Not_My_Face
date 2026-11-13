# tests/test_smoke.py
import torch
import sys
import os

# Add src to python path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models.dual_stream_net import DualStreamNet
from training.train import BCEWithLabelSmoothingLoss

def test_model_smoke():
    print("Running Model Smoke Test...")
    
    # 1. Setup device (CPU is fine for smoke test on local machine)
    device = torch.device("cpu")
    
    # 2. Initialize model
    model = DualStreamNet(num_classes=1, pretrained=False).to(device)
    model.eval()
    
    # 3. Create dummy batch (Batch size = 2, Channels = 3, Height = 256, Width = 256)
    dummy_rgb = torch.randn(2, 3, 256, 256, device=device)
    dummy_freq = torch.randn(2, 3, 256, 256, device=device)
    dummy_targets = torch.tensor([0.0, 1.0], device=device)
    
    # 4. Test forward pass
    try:
        outputs = model(dummy_rgb, dummy_freq)
        print(f"Forward pass successful! Output shape: {outputs.shape}")
        assert outputs.shape == (2, 1), f"Expected shape (2, 1), got {outputs.shape}"
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e
        
    # 5. Test loss calculation & backward pass
    criterion = BCEWithLabelSmoothingLoss(smoothing=0.1)
    loss = criterion(outputs, dummy_targets.unsqueeze(1))
    print(f"Loss calculation successful! Loss value: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass & gradient check successful!")
    print("✅ All smoke tests passed successfully!")

if __name__ == "__main__":
    test_model_smoke()