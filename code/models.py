import torch
import torch.nn as nn
import torchvision.models as models

# Definimos varios modelos para regresión basados en arquitecturas de redes neuronales convolucionales.


class MobileNetV2Regression(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Regression, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=pretrained)
        # Reemplazar la capa fully connected para regresión
        self.mobilenet_v2.classifier[1] = nn.Linear(self.mobilenet_v2.classifier[1].in_features, 1)

    def forward(self, x):
        x = self.mobilenet_v2(x)
        return x

class MobileNetV2Classification(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(MobileNetV2Classification, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=pretrained)
        # Reemplazar la capa fully connected para clasificación
        self.classes = num_classes
        self.mobilenet_v2.classifier[1] = nn.Linear(self.mobilenet_v2.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.mobilenet_v2(x)
        return x

#Modelo pequeño de clasificacion de 5 clases recibiendo imagen 224x224
class ResNet18Classification(nn.Module):
    def __init__(self, num_classes = 5, pretrained = True):
        super(ResNet18Classification, self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        # Reemplazar la capa fully connected para clasificación
        self.classes = num_classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x

class ResNet18Regression(nn.Module):
    def __init__(self,num_classes = 5, pretrained = True):
        super(ResNet18Regression, self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.classes = num_classes
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet18(x)
        return x

class ResNet50Classification(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50Classification, self).__init__()
        self.classes = num_classes
        self.resnet50 = models.resnet50(pretrained=pretrained)
        # Reemplazar la capa fully connected para clasificación
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x

class ResNet50Regression(nn.Module):
    def __init__(self,num_classes = 5, pretrained = True):

        super(ResNet50Regression, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        self.classes = num_classes
        # Reemplazar la capa fully connected para regresión
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet50(x)
        return x

class EfficientNetB0Classification(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(EfficientNetB0Classification, self).__init__()
        self.efficientnet_b0 = models.efficientnet_b0(pretrained=pretrained)
        # Reemplazar la capa fully connected para clasificación
        self.efficientnet_b0.classifier[1] = nn.Linear(self.efficientnet_b0.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.efficientnet_b0(x)
        return x

# Not tested under this context
class DenseNet121Classification(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(DenseNet121Classification, self).__init__()
        self.densenet121 = models.densenet121(pretrained=pretrained)
        # Reemplazar la capa fully connected para clasificación
        self.densenet121.classifier = nn.Linear(self.densenet121.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.densenet121(x)
        return x

class VisionTransformerClassification(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(VisionTransformerClassification, self).__init__()
        self.vit = models.vit_b_16(pretrained=pretrained)
        # Reemplazar la cabeza de clasificación
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x

class SwinTransformerClassification(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(SwinTransformerClassification, self).__init__()
        self.swin_t = models.swin_t(pretrained=pretrained)
        # Reemplazar la cabeza de clasificación
        self.swin_t.head = nn.Linear(self.swin_t.head.in_features, num_classes)

    def forward(self, x):
        x = self.swin_t(x)
        return x

def getModels():
    return {
        "MobileNetV2Regression": MobileNetV2Regression,
        "MobileNetV2Classification": MobileNetV2Classification,
        "ResNet18Classification": ResNet18Classification,
        "ResNet18Regression": ResNet18Regression,
        "ResNet50Classification": ResNet50Classification,
        "ResNet50Regression": ResNet50Regression,
        "EfficientNetB0Classification": EfficientNetB0Classification,
        "DenseNet121Classification": DenseNet121Classification,
        "VisionTransformerClassification": VisionTransformerClassification,
        "SwinTransformerClassification": SwinTransformerClassification
    }