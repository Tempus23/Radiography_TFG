import torch
import torch.nn as nn
from torchvision import models

class RadiographyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RadiographyClassifier, self).__init__()
        # Cargar el modelo preentrenado ResNet18
        self.model = models.resnet18(pretrained=True)
        # Congelar las capas preentrenadas¡
        for param in self.model.parameters():
            param.requires_grad = False
        # Reemplazar la última capa totalmente conectada
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Modelo de clasificación de imagenes desde cero
class RadiographyClassifierFromScratch(nn.Module):
    def __init__(self, num_classes):
        super(RadiographyClassifierFromScratch, self).__init__()
        # Definir la arquitectura de la red
        self.model = nn.Sequential(
            #Capa de entrada imagen 224x224
            nn.Conv2d(224, 8, kernel_size=224, padding=32),
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
