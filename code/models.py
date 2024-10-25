import torch
import torch.nn as nn
import torchvision.models as models


        
class CustomModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.classes = num_classes

        self._embeed = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) # 224
        )

        self._block1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(224, 224)
                )
                for i in range(1)
            ]
        )


        self._final = nn.Sequential(
            nn.Linear(64 * 224 * 224, num_classes)
        )

    def forward(self, x):
        x = self._embeed(x)

        for block in self._block1:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self._final(x)
        return x




def getModels():
    return {
        "test" : CustomModel,
    }