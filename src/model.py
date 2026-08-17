"""
model.py
========
Defines the image classification model.

get_model(architecture, num_classes) returns an nn.Module ready to train:
  - "resnet18" - torchvision's ResNet-18, trained from scratch (no pretrained
    weights), with its final fully-connected layer resized to num_classes.
  - "cnn"      - a small custom CNN, useful as a fast-to-train baseline.
"""

import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """A small 3-block CNN for 32x32 RGB images (CIFAR-10 sized)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(architecture: str, num_classes: int = 10) -> nn.Module:
    """Factory function - returns the requested model architecture."""
    architecture = architecture.lower()

    if architecture == "resnet18":
        model = models.resnet18(weights=None)  # trained from scratch
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "cnn":
        return SimpleCNN(num_classes=num_classes)

    raise ValueError(f"Unknown architecture: {architecture!r}. Expected 'resnet18' or 'cnn'.")