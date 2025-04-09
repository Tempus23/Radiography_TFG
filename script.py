import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.models.efficientnet import EfficientNetB5Custom
from src.grad_cam import GradCAM
model_state = torch.load(r'models\OAI Mendeley\best_model_EfficientNetB5Custom_epoch_2.pt',map_location=torch.device('cpu'), weights_only=False)
model = EfficientNetB5Custom(num_classes=5, pretrained=False)
model.load_state_dict(model_state)
model = torch.load(r'models\OAI Mendeley\best_model_EfficientNetB5_epoch_30.pt',map_location=torch.device('cpu'), weights_only=False)
model.eval()
target_layer = model.efficientnet.features[-1]

grad_cam = GradCAM(model, target_layer)

class gatosDataset(Dataset):
    def __init__(self, batch_size=32, transform=None, local=False, path="dataset/gatos/jpg"):
        if local:
            print("LOCAL MODE ENABLED")
        self.transform = transform
        self.data_path = path
        self.data = []
        self.batch_size = batch_size
        self.classes = sorted(os.listdir(self.data_path))  # Lista de clases
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_dataloader(self, shuffle=True):
        return DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)
    

    import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import *
from src.data import *
from src.models.efficientnet import *
from src.utils import *
from src.train import train, train_model, test_model
from src.trainers.classification import Classification
from src.trainers.regresion import Regression

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
class HistogramEqualization:
    """Aplica ecualización de histograma para ajuste de contraste"""
    def __call__(self, img):
        # Convertir PIL Image a numpy array
        img_np = np.array(img)
        
        # Aplicar ecualización de histograma por canal
        if len(img_np.shape) == 3:  # Imagen RGB
            img_eq = np.zeros_like(img_np)
            for i in range(3):
                img_eq[:,:,i] = cv2.equalizeHist(img_np[:,:,i])
        else:  # Imagen en escala de grises
            img_eq = cv2.equalizeHist(img_np)
            
        # Convertir de nuevo a PIL Image
        return Image.fromarray(img_eq)

class BilateralFilter:
    """Aplica filtrado bilateral para suavizado preservando bordes"""
    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = d  # Diámetro de cada vecindario de píxeles
        self.sigma_color = sigma_color  # Filtro sigma en el espacio de color
        self.sigma_space = sigma_space  # Filtro sigma en el espacio de coordenadas
    
    def __call__(self, img):
        # Convertir PIL Image a numpy array
        img_np = np.array(img)
        
        # Aplicar filtro bilateral
        img_filtered = cv2.bilateralFilter(
            img_np, self.d, self.sigma_color, self.sigma_space)
            
        # Convertir de nuevo a PIL Image
        return Image.fromarray(img_filtered)

transform =  transforms.Compose([
            transforms.Resize((224, 224)), #Normalizar 
            HistogramEqualization(),
            BilateralFilter(),
            transforms.ToTensor(),
        ])
BATCH_SIZE = 10
LEARNING_RATE = 0.001
FACTOR = 0.001
L1 = 0.00
L2 = 0.00
PATIENCE = 5
BETAS=(0.9, 0.999)
# Regularización L1 y L2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


val_dataset = gatosDataset(batch_size=BATCH_SIZE, transform=transform)
# Mosrar primera imagen
img, label = val_dataset[0]
plt.imshow(img.permute(1, 2, 0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Regression(model, device, L1=L1, L2=L2, lr=LEARNING_RATE, factor=FACTOR, patience=PATIENCE, betas=BETAS)
train_model(model,trainer,train_dataset=val_dataset, val_dataset=val_dataset, epochs=10, wdb=False, plot_loss=True, device=device)
test_model(model,val_dataset.get_dataloader(),trainer,device)