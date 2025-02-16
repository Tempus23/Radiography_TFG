import os
from tabulate import tabulate

def explorar_split_data(path):
    imagenes = {}
    for subset in ["train", "val", "test"]:
        subset_path = os.path.join(path, subset)
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