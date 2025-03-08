import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from PIL import Image
import cv2
import numpy as np
from src.config import *  # Asegúrate de que MENDELEY_OAI_224_SPLIT_PATH está bien definido
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def dataset_augmentation(ORIGINAL_PATH, NEW_PATH, max_images=3000):
  
    ORIGINAL_TRAIN_PATH = os.path.join(ORIGINAL_PATH, 'train')
    NEW_TRAIN_PATH = os.path.join(NEW_PATH, 'train')
    NEW_VAL_PATH = os.path.join(NEW_PATH, 'val')
    NEW_TEST_PATH = os.path.join(NEW_PATH, 'test')
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest'
    )
    AUG_IMAGES = max_images
    
    if not os.path.exists(NEW_PATH):
        os.makedirs(NEW_PATH)
        os.makedirs(NEW_TRAIN_PATH)
        os.makedirs(NEW_VAL_PATH)
        os.makedirs(NEW_TEST_PATH)

    # Data augmentation in train
    classes = os.listdir(ORIGINAL_TRAIN_PATH)
    for class_name in classes:
        CLASS_PATH = os.path.join(NEW_TRAIN_PATH, class_name)
        class_origin = os.path.join(ORIGINAL_TRAIN_PATH, class_name)
        num_augmentations = AUG_IMAGES - len(os.listdir(class_origin))

        # Copy original images
        if not os.path.exists(CLASS_PATH):
            os.makedirs(CLASS_PATH)
        
        print("Copying original images...")
        for img_name in os.listdir(class_origin):
            img_path = os.path.join(class_origin, img_name)
            img = cv2.imread(img_path)
            
            # Verificar si la imagen fue leída correctamente
            if img is None:
                print(f"Error al leer la imagen {img_path}. Puede que no sea una imagen válida o esté dañada.")
                continue
            
            
            # Copiar la imagen
            new_img_path = os.path.join(CLASS_PATH, f"{class_name}_{img_name}")
            cv2.imwrite(new_img_path, img)

        print(f"Se han copiado {len(os.listdir(CLASS_PATH))} imágenes de la clase {class_name}")


        print(f"Generando {num_augmentations} imágenes aumentadas para la clase {class_name}...")
        while(len(os.listdir(CLASS_PATH)) < AUG_IMAGES):
            
            probabilidad = ((AUG_IMAGES - len(os.listdir(CLASS_PATH))) / len(os.listdir(class_origin))) + 0.05
            print(f"Probabilidad de que se genere una imagen aumentada: {probabilidad}")
            for img_name in os.listdir(class_origin):
                if len(os.listdir(CLASS_PATH)) >= AUG_IMAGES:
                    break
                if np.random.rand() > probabilidad:
                    continue
                
                img_path = os.path.join(class_origin, img_name)
                img = cv2.imread(img_path)
                img_array = img.reshape((1, ) + img.shape)
                for batch in data_gen.flow(img_array, batch_size=1, save_to_dir=CLASS_PATH, save_prefix='aug', save_format='png'):
                    break
                
                

        print(f"Se han generado {len(os.listdir(CLASS_PATH))} imágenes aumentadas para la clase {class_name}\n-----------------------------------\n")

    # Copy val and test images

    for mode in ['val', 'test']:
        ORIGINAL_MODE_PATH = os.path.join(ORIGINAL_PATH, mode)
        NEW_MODE_PATH = os.path.join(NEW_PATH, mode)
        for class_name in classes:
            CLASS_PATH = os.path.join(NEW_MODE_PATH, class_name)
            class_origin = os.path.join(ORIGINAL_MODE_PATH, class_name)
            if not os.path.exists(CLASS_PATH):
                os.makedirs(CLASS_PATH)
            
            for img_name in os.listdir(class_origin):
                img_path = os.path.join(class_origin, img_name)
                img = cv2.imread(img_path)
                
                # Verificar si la imagen fue leída correctamente
                if img is None:
                    print(f"Error al leer la imagen {img_path}. Puede que no sea una imagen válida o esté dañada.")
                    continue
                
                # Copiar la imagen
                new_img_path = os.path.join(CLASS_PATH, f"{class_name}_{img_name}")
                cv2.imwrite(new_img_path, img)
            print(f"Se han copiado {len(os.listdir(CLASS_PATH))} imágenes de la clase {class_name} en el conjunto {mode}\n-----------------------------------\n")


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

class DatasetExperiment1(Dataset):
    def __init__(self, mode='train', batch_size=32,grey = False, local = False, path = ''):
        """
        Args:
            mode (str): 'train', 'val' o 'test'.
            transform: Transformaciones de torchvision a aplicar a las imágenes.
        """
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'"
        if local:
            print("LOCAL MODE ENABLED")
        self.grey = grey
        # Transformaciones del paper
        # Histogram equalization for contrast adjustment
        # and bilateral filtering for smoothness
        self.transform =  transforms.Compose([
            transforms.Resize((224, 224)),
            HistogramEqualization(),
            BilateralFilter(d=9, sigma_color=75, sigma_space=75),
            transforms.ToTensor(),
        ])
        self.data_path = os.path.join(path, mode)
        self.classes = sorted(os.listdir(self.data_path))  # Lista de clases
        self.data = []
        self.batch_size = batch_size
        # Cargar imágenes con sus etiquetas
        

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            i = 0
            for img_name in os.listdir(class_path):
                if local and i >= 3:
                    break
                img_path = os.path.join(class_path, img_name)
                self.data.append((img_path, label))
                i += 1
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        if not self.grey:
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_dataloader(self, shuffle=True):       
        return DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)




class OriginalOAIDataset(Dataset):
    def __init__(self, mode='train', batch_size=32, transform=None, local = False, path = MENDELEY_OAI_224_SPLIT_PATH):
        """
        Args:
            mode (str): 'train', 'val' o 'test'.
            transform: Transformaciones de torchvision a aplicar a las imágenes.
        """
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'"
        
        if local:
            print("LOCAL MODE ENABLED")
        self.transform = transform
        self.data_path = os.path.join(path, mode)
        self.classes = sorted(os.listdir(self.data_path))  # Lista de clases
        self.data = []
        self.batch_size = batch_size
        # Cargar imágenes con sus etiquetas
        

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            i = 0
            for img_name in os.listdir(class_path):
                if local and i >= 3:
                    break
                img_path = os.path.join(class_path, img_name)
                self.data.append((img_path, label))
                i += 1
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


