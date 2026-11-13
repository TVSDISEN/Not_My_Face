# tests/test_smoke.py
import torch
import unittest

from src.models.dual_stream_net import DualStreamNet
from src.training.train import BCEWithLabelSmoothingLoss


class ModelSmokeTests(unittest.TestCase):
    def test_forward_loss_and_backward(self):
        device = torch.device("cpu")
        model = DualStreamNet(num_classes=1, pretrained=False).to(device)
        model.eval()
        rgb = torch.randn(2, 3, 256, 256, device=device)
        frequency = torch.randn(2, 3, 256, 256, device=device)
        targets = torch.tensor([0.0, 1.0], device=device).unsqueeze(1)

        logits = model(rgb, frequency)
        self.assertEqual(tuple(logits.shape), (2, 1))

        loss = BCEWithLabelSmoothingLoss(smoothing=0.1)(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()