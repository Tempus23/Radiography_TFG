from torch import nn, Tensor
import torch
from typing import Optional
import timm

class EfficientNetB5(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super(EfficientNetB5, self).__init__()
        # Cargamos EfficientNetB5 preentrenado sin la cabeza clasificadora
        # (num_classes=0 desactiva la capa final, devolviendo el vector de características)
        self.base_model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=0)
        
        # Si el número de canales de entrada es distinto de 3, se debe ajustar la capa inicial
        if in_channels != 3:
            conv_stem = self.base_model.conv_stem
            self.base_model.conv_stem = nn.Conv2d(
                in_channels,
                conv_stem.out_channels,
                kernel_size=conv_stem.kernel_size,
                stride=conv_stem.stride,
                padding=conv_stem.padding,
                bias=conv_stem.bias is not None
            )
        
        # Obtiene la dimensión de salida del extractor de características
        # En timm, EfficientNetB5 finaliza con 'classifier' que se puede usar para conocer la dimensión
        in_features = self.base_model.num_features if hasattr(self.base_model, 'num_features') else 2048
        
        # Definimos una nueva cabeza clasificadora
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        # Extraemos las características con EfficientNetB5
        x = self.base_model(x)
        # Aplicamos dropout para regularización
        x = self.dropout(x)
        # Clasificamos con la nueva capa fully-connected
        x = self.fc(x)
        return x

# Ejemplo de uso:
if __name__ == '__main__':
    model = EfficientNetB5(num_classes=10)
    print(model)
    
    # Simulamos una entrada (típicamente EfficientNetB5 espera imágenes de 456x456)
    x = torch.randn(1, 3, 456, 456)
    out = model(x)
    print(out.shape)  # Debe ser [1, 10]
