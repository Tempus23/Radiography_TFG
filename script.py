import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np

# Ruta al directorio donde están las imágenes DICOM
dicom_dir = "dataset/gatos/Normal"

def load_dicom_images(directory, num = None):
    """Carga todas las imágenes DICOM de un directorio."""
    dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    dicom_images = []
    for i, file in enumerate(dicom_files):
        if num is not None and i >= num:
            break
        try:
            ds = pydicom.dcmread(file)
            dicom_images.append(ds)
        except Exception as e:
            print(f"Error al cargar {file}: {e}")
    return dicom_images

def display_dicom_image(dicom_image):
    """Muestra una imagen DICOM usando matplotlib."""
    plt.imshow(dicom_image.pixel_array, cmap=plt.cm.gray)
    plt.title(f"Paciente: {dicom_image.PatientID}")
    plt.axis('off')
    plt.show()
def print_dicom_clean_metadata(dicom_image):
    """Imprime los metadatos DICOM, excluyendo los datos binarios (como Pixel Data)."""
    for elem in dicom_image:
        if elem.tag == (0x7FE0, 0x0010):  # Pixel Data
            continue  # O puedes poner print("Pixel Data: [omitido]")
        if elem.VR == 'OB' or elem.VR == 'OW' or elem.VR == 'UN':
            print(f"{elem.tag} {elem.name}: [Tipo binario omitido]")
        else:
            print(f"{elem.tag} {elem.name}: {elem.value}")
def compare_dicom_metadata(dcm1, dcm2):
    """Compara dos objetos DICOM y muestra solo los campos que son distintos."""
    print("\n📌 Comparando metadatos...")
    tags1 = {elem.tag: elem for elem in dcm1 if elem.tag != (0x7FE0, 0x0010)}
    tags2 = {elem.tag: elem for elem in dcm2 if elem.tag != (0x7FE0, 0x0010)}
    
    all_tags = set(tags1.keys()).union(tags2.keys())

    for tag in sorted(all_tags):
        val1 = tags1.get(tag)
        val2 = tags2.get(tag)

        if val1 is None or val2 is None:
            print(f"{tag} sólo presente en una imagen")
        elif val1.value != val2.value:
            print(f"{tag} {val1.name}:")
            print(f"    Imagen 1: {val1.value}")
            print(f"    Imagen 2: {val2.value}")

dicom_images = load_dicom_images(dicom_dir,num=3)

# Visualizar las primeras imágenes
print_dicom_clean_metadata(dicom_images[1])    