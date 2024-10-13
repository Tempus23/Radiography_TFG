import os
import sys
sys.path.append("..")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from definitions import BATCH_SIZE, RUTA_DATOS

transformaciones_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transformaciones_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_train = datasets.ImageFolder(
    root=os.path.join(RUTA_DATOS, 'train'),
    transform=transformaciones_train
)

dataset_val = datasets.ImageFolder(
    root=os.path.join(RUTA_DATOS, 'val'),
    transform=transformaciones_val
)

dataset_test = datasets.ImageFolder(
    root=os.path.join(RUTA_DATOS, 'test'),
    transform=transformaciones_val
)


loader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2 # Ajusta según tu sistema
)

loader_val = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

loader_test = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

