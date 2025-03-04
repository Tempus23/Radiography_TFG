import os
from tabulate import tabulate
import cv2
import matplotlib.pyplot as plt

def explorar_split_data(path):
    imagenes = {}
    for subset in ["train", "val", "test", "auto_test"]:
        subset_path = os.path.join(path, subset)
        if not os.path.exists(subset_path):
            continue
        if os.path.exists(subset_path):
            imagenes[subset] = []  # Inicializamos la lista para cada subset
            for root, dirs, files in os.walk(subset_path):
                # Omitimos el directorio raíz
                if root == subset_path:
                    continue
                imagenes[subset].append(len(files))
    print_split_table(imagenes)
    return imagenes

def explorar_data(path):
    imagenes = []
    for root, dirs, files in os.walk(path):
        # Omitimos el directorio raíz
        if root == path:
            continue
        imagenes.append(len(files))
    print_table(imagenes)
    return imagenes

def print_table(data):
    tabla = []

    for i in range(len(data)):
        fila = [f"{i}", data[i]]
        tabla.append(fila)

    print(tabulate(tabla, headers=["Clase", "Cantidad"], tablefmt="fancy_grid"))

def print_split_table(data):
    num_clases = len(data['train'])
    tabla = []

    for i in ["train", "val", "test"]:
        fila = [f"{i}", data[i][0], data[i][1], data[i][2], data[i][3], data[i][4]]
        tabla.append(fila)

    print(tabulate(tabla, headers=["Clase", "0", "1", "2", "3", "4"], tablefmt="fancy_grid"))

def mostrar_imagenes(directorio_principal, imagenes_por_clase=4):
    """
    Muestra un número específico de imágenes de cada subdirectorio dentro del directorio principal.

    Parámetros:
    directorio_principal (str): Ruta al directorio principal que contiene los subdirectorios.
    imagenes_por_subdirectorio (int): Número de imágenes a mostrar por cada subdirectorio. Por defecto es 2.
    """
    # Verificar si el directorio principal existe
    if not os.path.isdir(directorio_principal):
        print(f"El directorio {directorio_principal} no existe.")
        return

    # Obtener la lista de subdirectorios en el directorio principal
    subdirectorios = [d for d in os.listdir(directorio_principal) if os.path.isdir(os.path.join(directorio_principal, d))]

    # Inicializar listas para almacenar imágenes y títulos
    imagenes = []
    titulos = []

    # Recorrer cada subdirectorio
    for subdirectorio in subdirectorios:
        ruta_subdirectorio = os.path.join(directorio_principal, subdirectorio)
        # Obtener la lista de archivos en el subdirectorio
        archivos = os.listdir(ruta_subdirectorio)
        # Filtrar solo los archivos que son imágenes (por extensión)
        extensiones_permitidas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        imagenes_subdir = [f for f in archivos if f.lower().endswith(extensiones_permitidas)]

        # Seleccionar las primeras 'imagenes_por_subdirectorio' imágenes
        for img_name in imagenes_subdir[:imagenes_por_clase]:
            img_path = os.path.join(ruta_subdirectorio, img_name)
            # Leer la imagen
            img = cv2.imread(img_path)
            # Verificar si la imagen fue leída correctamente
            if img is None:
                print(f"Error al leer la imagen {img_path}.")
                continue
            # Convertir la imagen de BGR a RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Agregar la imagen y su título a las listas
            imagenes.append(img_rgb)
            titulos.append(f'{subdirectorio}: {img_name}')

    # Verificar si se encontraron imágenes
    if not imagenes:
        print("No se encontraron imágenes en los subdirectorios.")
        return

    # Definir el número de columnas para la visualización
    columnas = 10
    filas = (len(imagenes) + columnas - 1) // columnas  # Calcular el número de filas necesarias

    # Crear la figura y los ejes
    fig, axs = plt.subplots(filas, columnas, figsize=(15, 5 * filas))
    axs = axs.flatten()  # Aplanar la matriz de ejes para iterar fácilmente

    # Mostrar cada imagen en su subplot correspondiente
    for ax, img, titulo in zip(axs, imagenes, titulos):
        ax.imshow(img)
        ax.set_title(titulo)
        ax.axis('off')

    # Desactivar los ejes no utilizados
    for ax in axs[len(imagenes):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
