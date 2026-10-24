"""
model.py
MobileNetV2 transfer learning model for Fruit Quality Inspection (6 classes).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FruitMobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super(FruitMobileNetV2, self).__init__()

        if pretrained:
            weights = models.MobileNet_V2_Weights.DEFAULT
            self.backbone = models.mobilenet_v2(weights=weights)
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        # Replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
