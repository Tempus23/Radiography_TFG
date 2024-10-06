import os
import shutil
import random

def dividir_dataset(ruta_dataset, ruta_destino, porcentaje_val=0.15, porcentaje_test=0.15):
    # Crear directorios de destino si no existen
    conjuntos = ['train', 'val', 'test']
    clases = [d for d in os.listdir(ruta_dataset) if os.path.isdir(os.path.join(ruta_dataset, d))]
    
    
    for conjunto in conjuntos:
        for clase in clases:
            ruta_clase_destino = os.path.join(ruta_destino, conjunto, clase)
            os.makedirs(ruta_clase_destino, exist_ok=True)
    
    # Dividir y copiar las imágenes
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

# Establecer la semilla para reproducibilidad
random.seed(39)

ruta_dataset = 'dataset/data'  # Dataset Experto 1
ruta_destino = 'dataset/split_data'  # Reemplaza con la ruta donde quieres guardar el dataset dividido

dividir_dataset(ruta_dataset, ruta_destino, porcentaje_val=0.15, porcentaje_test=0.15)
