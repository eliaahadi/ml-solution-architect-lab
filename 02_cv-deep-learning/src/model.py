import torch
from torch import nn

from . import config


class CnnClassifier(nn.Module):
    """Simple CNN for CIFAR-10-like images (3x32x32)."""

    def __init__(self, num_classes: int = config.num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(device: torch.device | None = None) -> nn.Module:
    """Create model and move it to device (cpu/cuda)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnClassifier(num_classes=config.num_classes).to(device)
    return model