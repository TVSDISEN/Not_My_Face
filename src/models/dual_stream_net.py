# src/models/dual_stream_net.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class FrequencyBranch(nn.Module):
    """Processes frequency domain input (FFT/Log-Amplitude spectrum)."""
    def __init__(self, output_dim=256):
        super().__init__()
        # Lightweight CNN to extract frequency-level artifacts
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DualStreamNet(nn.Module):
    """
    Dual-Stream Network combining a Spatial Branch (EfficientNet) 
    and a Frequency Branch for Deepfake Detection.
    """
    def __init__(self, num_classes=1, pretrained=True, dropout_rate=0.4):
        super().__init__()
        
        # 1. Spatial Branch: EfficientNet-B2
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.spatial_backbone = efficientnet_b2(weights=weights)
        
        # Extract spatial feature dimension before final classifier
        spatial_out_dim = self.spatial_backbone.classifier[1].in_features
        self.spatial_backbone.classifier = nn.Identity() # Remove default classifier
        
        # 2. Frequency Branch
        self.frequency_branch = FrequencyBranch(output_dim=256)
        
        # 3. Fusion & Classifier Head
        combined_dim = spatial_out_dim + 256
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(combined_dim, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_image, freq_image):
        """
        Args:
            rgb_image (torch.Tensor): RGB face crop [B, 3, H, W]
            freq_image (torch.Tensor): Frequency spectrum map [B, 3, H, W]
        """
        # Extract spatial features
        spatial_feat = self.spatial_backbone(rgb_image)
        
        # Extract frequency features
        freq_feat = self.frequency_branch(freq_image)
        
        # Concatenate both streams
        combined = torch.cat((spatial_feat, freq_feat), dim=1)
        
        # Final classification logits
        out = self.classifier(combined)
        return out