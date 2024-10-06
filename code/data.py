import os
import sys
sys.path.append("..")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from definitions import BATCH_SIZE, RUTA_DATOS


transformaciones_train = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomRotation(10),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Ajusta según tus necesidades
])

transformaciones_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

contrast_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.2, contrast=2),
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
    num_workers=4 # Ajusta según tu sistema
)

loader_val = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

loader_test = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

