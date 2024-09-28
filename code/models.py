import torch
import torch.nn as nn
from torchvision import models

class RadiographyClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RadiographyClassifier, self).__init__()
        # Cargar el modelo preentrenado ResNet18
        self.model = models.resnet18(pretrained=True)
        # Congelar las capas preentrenadas
        #Ajustar primera capa para aceptar 2 imagenes
        self.model.conv1 = nn.Conv2d(6, self.model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        # Reemplazar la última capa totalmente conectada
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.model(x)