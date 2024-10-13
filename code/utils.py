import matplotlib.pyplot as plt
import numpy as np

class2name = ['Normal','Leve','Moderado','Grave','Muy Grave']

def show_img(tensor, label = ""):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Eliminar la dimensión del lote si está presente
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convertir el tensor a numpy y transponer las dimensiones de (C, H, W) a (H, W, C)
    image = tensor.permute(1, 2, 0).numpy()
    
    # Desnormalizar la imagen si fue normalizada (opcional, dependiendo de tu preprocesamiento)
    # image = image * 0.5 + 0.5
    
    
    plt.imshow(image)
    plt.title(class2name[label.numpy()])
    plt.axis('off')  # Ocultar los ejes
    plt.show()