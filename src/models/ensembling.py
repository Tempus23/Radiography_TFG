from torch import nn, Tensor
import torch
from typing import Type
from .efficientnet import EfficientNetB0, EfficientNetB4

# Clase Ensembling
class Ensembling(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super(Ensembling, self).__init__()
        self.name = "EnsemblingFaruq"
        self.efficientnetb0 = EfficientNetB0(num_classes=num_classes, pretrained=pretrained)
        self.efficientnetb4 = EfficientNetB4(num_classes=num_classes, pretrained=pretrained)

        self.b0_features = 1280
        self.b4_features = 1790


        #Clasificación 512 -> 256 -> 128 -> 65 -> num_classes con batchNormalization, l2regularization y dropout
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(150528),

            nn.Linear(150528, 512),
            nn.ReLU(),

            nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:

        # Parte 1 - Features from efficientNet b0 and b4
        features_b0 = self.efficientnetb0.features(x)
        features_b4 = self.efficientnetb4.features(x)

        # Flatten
        features_b0 = torch.flatten(features_b0, 1)
        features_b4 = torch.flatten(features_b4, 1)

        #Concatenar
        x = torch.cat((features_b0, features_b4), dim=1)

        #Clasificación 512 -> 256 -> 128 -> 65 -> num_classes con batchNormalization, l2regularization y dropout
        x = self.classifier(x)

        return x