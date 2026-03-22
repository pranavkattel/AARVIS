import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceEmbeddingModel(nn.Module):
    """Runtime face embedding backbone used by the server."""

    def __init__(self, embedding_size: int = 512, dropout: float = 0.45, unfreeze_from: int = 10):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for i, layer in enumerate(self.backbone):
            if i < unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(1280, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)
