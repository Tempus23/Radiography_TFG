import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

class ResNet50Model(nn.Module):
    """
    Modelo basado en ResNet50 para la clasificación de radiografías.
    Permite personalización en el número de clases y si se usa transfer learning.
    """
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=False, dropout_rate=0.3):
        """
        Inicializa el modelo ResNet50.
        
        Args:
            num_classes (int): Número de clases para la clasificación (default: 5)
            pretrained (bool): Si se deben usar pesos preentrenados en ImageNet (default: True)
            freeze_backbone (bool): Si se deben congelar las capas de la red base (default: False)
            dropout_rate (float): Tasa de dropout aplicada antes de la capa de clasificación (default: 0.3)
        """
        super(ResNet50Model, self).__init__()
        
        # Cargar el modelo base ResNet50
        self.model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Nombre del modelo para identificación
        self.name = "ResNet50"
        
        # Congelar los parámetros de la red si se especifica
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Reemplazar la capa de clasificación final
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Propagación hacia adelante a través del modelo.
        
        Args:
            x: Tensor de entrada con forma [batch_size, channels, height, width]
            
        Returns:
            Tensor con las predicciones de clase [batch_size, num_classes]
        """
        return self.model(x)
    
    def get_features(self, x):
        """
        Obtiene los features del modelo antes de la capa de clasificación.
        Útil para análisis de características o transferencia de estilo.
        
        Args:
            x: Tensor de entrada con forma [batch_size, channels, height, width]
            
        Returns:
            Tensor de características [batch_size, 2048]
        """
        # Extrae todas las capas excepto la final
        modules = list(self.model.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        
        # Obtiene los features y los aplana
        features = feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        return features

class ResNet50WithAttention(nn.Module):
    """
    Modelo ResNet50 con mecanismo de atención adicional para
    resaltar regiones relevantes en las radiografías.
    """
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=False, dropout_rate=0.3):
        """
        Inicializa el modelo ResNet50 con atención.
        
        Args:
            num_classes (int): Número de clases para la clasificación (default: 5)
            pretrained (bool): Si se deben usar pesos preentrenados en ImageNet (default: True)
            freeze_backbone (bool): Si se deben congelar las capas de la red base (default: False)
            dropout_rate (float): Tasa de dropout aplicada antes de la capa de clasificación (default: 0.3)
        """
        super(ResNet50WithAttention, self).__init__()
        
        # Cargar el modelo base ResNet50
        self.model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Nombre del modelo para identificación
        self.name = "ResNet50WithAttention"
        
        # Eliminar la capa de clasificación final
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Definir el mecanismo de atención
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Capas de clasificación final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Propagación hacia adelante con mecanismo de atención.
        
        Args:
            x: Tensor de entrada con forma [batch_size, channels, height, width]
            
        Returns:
            logits: Tensor con las predicciones de clase [batch_size, num_classes]
            attention_map: Mapa de atención para visualización (opcional)
        """
        # Extraer características
        features = self.features(x)
        
        # Calcular mapa de atención
        attention_map = self.attention(features)
        
        # Aplicar atención a las características
        attended_features = features * attention_map
        
        # Pooling global y clasificación
        x = self.avgpool(attended_features)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        
        # Devolver ambos para poder visualizar la atención si se desea
        return logits, attention_map
    
    def get_attention_maps(self, x):
        """
        Obtiene solo los mapas de atención para visualización.
        
        Args:
            x: Tensor de entrada con forma [batch_size, channels, height, width]
            
        Returns:
            attention_map: Mapa de atención normalizado
        """
        features = self.features(x)
        attention_map = self.attention(features)
        return attention_map

