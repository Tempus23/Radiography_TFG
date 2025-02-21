from torch import nn, Tensor
import torch
from typing import Optional
import timm
from torchvision import models
from torchvision.models import EfficientNet_B5_Weights

class EfficientNetB5Custom(nn.Module):
    def __init__(self, num_classes=5, pretrained = True):
        super(EfficientNetB5Custom, self).__init__()
        # Cargar el modelo EfficientNetB5 preentrenado
        if pretrained:
            self.efficientnet = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b5(weights=None)
        self.name = "EfficientNetB5Custom"
        # Reemplazar la capa final del clasificador
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.efficientnet(x)