from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Bloque convolucional básico
        self.name = "SimpleCNN"
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.F = nn.functional
        # Capas totalmente conectadas
        self.fc1 = nn.Linear(16 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Primera convolución
        x = self.pool(self.F.relu(self.conv1(x)))

        # Flatten
        x = x.view(-1, 16 * 112 * 112)
        # Primera capa totalmente conectada
        x = self.F.relu(self.fc1(x))
        # Capa de salida
        x = self.fc2(x)
        x = self.F.softmax(x, dim=1)
        return x