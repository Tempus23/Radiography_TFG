import numpy as np
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.hook_layers()

    def hook_layers(self):
        """Registra hooks en la capa objetivo para obtener activaciones y gradientes."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]  # Guardar gradientes

        def forward_hook(module, input, output):
            self.activations = output  # Guardar activaciones

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        """Genera el mapa de activación Grad-CAM."""
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        self.model.zero_grad()
        output[:, class_idx].backward()

        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()

        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * activations, axis=1)

        cam = np.maximum(cam, 0)  # ReLU
        cam = cam[0] / np.max(cam)  # Normalizar

        return cam
