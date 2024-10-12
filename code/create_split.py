import os
import shutil
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from utils import show_img
from keras import preprocessing 
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

#Mostrar directorio actual
DATASET_PATH = 'dataset/data'
PROCESSED_DATA_PATH = 'dataset/processed_data'
SPLIT_PATH = 'dataset/split'
IMG_SIZE = (224, 224)
# Establecer la semilla para reproducibilidad
random.seed(39)

# Crear directorios de destino si no existen
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# Directorios para cada clase
classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
for c in classes:
    path = os.path.join(PROCESSED_DATA_PATH, c)
    if not os.path.exists(path):
        os.makedirs(path)

# # Preprocesar imagenes (Aumentar contraste, reducir ruido, normalizar tamaño)
# for class_name in classes:
#     class_dir = DATASET_PATH + '/' + class_name
#     split_class_dir = os.path.join(PROCESSED_DATA_PATH, class_name)

#     # Preprocess images
#     for img_name in os.listdir(class_dir):
#         img_path = os.path.join(class_dir, img_name)
#         try:
#             img = cv2.imread(img_path)

#             # Normalizar tamaño
#             img = cv2.resize(img, IMG_SIZE)
            
#             # Filtro de reducción de ruido (filtro gaussiano)
#             img = cv2.GaussianBlur(img, (5, 5), 0)

#             # Mejora del contraste utilizando la equalización del histograma en el canal YCrCb
#             ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#             ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
#             img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

#              # Guardar imagen procesada
#             cv2.imwrite(os.path.join(split_class_dir, img_name), img)
#         except:
#             print(f'Error al leer la imagen {img_path}')
#             continue

# Crear conjuntos de entrenamiento, validación y prueba
porcentaje_val = 0.15
porcentaje_test = 0.15

def crear_directorios_destino(ruta_destino, conjuntos, clases):
    for conjunto in conjuntos:
        for clase in clases:
            ruta_clase_destino = os.path.join(ruta_destino, conjunto, clase)
            os.makedirs(ruta_clase_destino, exist_ok=True)

conjuntos = ['train', 'val', 'test']
crear_directorios_destino(SPLIT_PATH, conjuntos, classes)
print("Dividiendo el dataset en conjuntos de entrenamiento, validación y prueba...")
def dividir_dataset(ruta_dataset, ruta_destino, porcentaje_val=0.15, porcentaje_test=0.15):
    clases = [d for d in os.listdir(ruta_dataset) if os.path.isdir(os.path.join(ruta_dataset, d))]
    
    for clase in clases:
        ruta_clase_origen = os.path.join(ruta_dataset, clase)
        imagenes = [f for f in os.listdir(ruta_clase_origen) if os.path.isfile(os.path.join(ruta_clase_origen, f))]
        random.shuffle(imagenes)
        
        num_total = len(imagenes)
        num_val = int(num_total * porcentaje_val)
        num_test = int(num_total * porcentaje_test)
        num_train = num_total - num_val - num_test
        
        imagenes_train = imagenes[:num_train]
        imagenes_val = imagenes[num_train:num_train+num_val]
        imagenes_test = imagenes[num_train+num_val:]
        
        # Copiar imágenes a las carpetas correspondientes
        for imagen in imagenes_train:
            ruta_origen = os.path.join(ruta_clase_origen, imagen)
            ruta_destino_train = os.path.join(ruta_destino, 'train', clase, imagen)
            shutil.copyfile(ruta_origen, ruta_destino_train)
        
        for imagen in imagenes_val:
            ruta_origen = os.path.join(ruta_clase_origen, imagen)
            ruta_destino_val = os.path.join(ruta_destino, 'val', clase, imagen)
            shutil.copyfile(ruta_origen, ruta_destino_val)
        
        for imagen in imagenes_test:
            ruta_origen = os.path.join(ruta_clase_origen, imagen)
            ruta_destino_test = os.path.join(ruta_destino, 'test', clase, imagen)
            shutil.copyfile(ruta_origen, ruta_destino_test)

dividir_dataset(PROCESSED_DATA_PATH, SPLIT_PATH, porcentaje_val, porcentaje_test)


# Aumentar datos
from torchvision import datasets, transforms
from torchvision.transforms import v2

data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_dir = os.path.join(SPLIT_PATH, "train")

print("Generando imágenes aumentadas...")
# Guardar imágenes aumentadas en los directorios de entrenamiento
for class_name in classes:
    class_dir = os.path.join(train_dir, class_name)
    num_augmentations = 1 + classes.index(class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        
        # Verificar si la imagen fue leída correctamente
        if img is None:
            print(f"Error al leer la imagen {img_path}. Puede que no sea una imagen válida o esté dañada.")
            continue
        
        # Convertir la imagen a un numpy array
        img_array = np.array(img)
        img_array = img_array.reshape((1,) + img_array.shape)  # Añadir dimensión batch
        # Generar imágenes aumentadas
        for i in range(num_augmentations):
            for batch in data_gen.flow(img_array, batch_size=1, save_to_dir=class_dir, save_prefix='aug', save_format='png'):
                break
    print(f"Se han generado {num_augmentations} imágenes aumentadas para la clase {class_name}")
  