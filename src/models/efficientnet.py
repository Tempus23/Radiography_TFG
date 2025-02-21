from torch import nn, Tensor
import torch
from typing import Optional
from torchvision import models
from torchvision.models import *

# Clase EfficientNetB5Custom
class EfficientNetB5Custom(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetB5Custom, self).__init__()
        if pretrained:
            self.efficientnet = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b5(weights=None)
        self.name = "EfficientNetB5Custom"
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

# Clase EfficientNetB5
class EfficientNetB5(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetB5, self).__init__()
        if pretrained:
            self.efficientnet = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b5(weights=None)
        self.name = "EfficientNetB5"
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)

# Clase EfficientNetB0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetB0, self).__init__()
        if pretrained:
            self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b0(weights=None)
        self.name = "EfficientNetB0"
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)

# Clase EfficientNetB4
class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetB4, self).__init__()
        if pretrained:
            self.efficientnet = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b4(weights=None)
        self.name = "EfficientNetB4"
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)

# Clase EfficientNetB7
class EfficientNetB7(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetB7, self).__init__()
        if pretrained:
            self.efficientnet = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        else:
            self.efficientnet = models.efficientnet_b7(weights=None)
        self.name = "EfficientNetB7"
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)
