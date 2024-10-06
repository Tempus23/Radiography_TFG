import torch
import torch.nn as nn
import torchvision.models as models

# Definimos varios modelos para regresión basados en arquitecturas de redes neuronales convolucionales.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 1)  # Salida de un solo valor para regresión

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet18Regression(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Regression, self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        # Reemplazar la capa fully connected para regresión
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class MobileNetV2Regression(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Regression, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=pretrained)
        # Reemplazar la capa fully connected para regresión
        self.mobilenet_v2.classifier[1] = nn.Linear(self.mobilenet_v2.classifier[1].in_features, 1)

    def forward(self, x):
        x = self.mobilenet_v2(x)
        return x


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Salida de un solo valor para regresión
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.regressor(x)
        return x


# Inicialización de los diferentes modelos
def get_model(model_name="simple_cnn"):
    if model_name == "simple_cnn":
        return SimpleCNN()
    elif model_name == "resnet18":
        return ResNet18Regression(pretrained=True)
    elif model_name == "mobilenet_v2":
        return MobileNetV2Regression(pretrained=True)
    elif model_name == "custom_cnn":
        return CustomCNN()
    else:
        raise ValueError(f"Modelo {model_name} no está disponible. Por favor elige entre 'simple_cnn', 'resnet18', 'mobilenet_v2', 'custom_cnn'.")


# Ejemplo de uso
if __name__ == "__main__":
    model_name = "resnet18"  # Puedes cambiar a simple_cnn, mobilenet_v2, custom_cnn
    model = get_model(model_name)
    print(model)