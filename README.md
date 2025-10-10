# 🦴 TFG: Clasificación de Artrosis de Rodilla mediante Redes Convolucionales Profundas

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red.svg)](https://pytorch.org/)

Este proyecto forma parte del **Trabajo de Fin de Grado (TFG)** en Ingeniería Informática en la **Universidad Politécnica de Valencia** y tiene como objetivo el desarrollo, entrenamiento y evaluación de modelos de deep learning para la clasificación automática del grado de artrosis de rodilla a partir de radiografías.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Tecnologías y Herramientas](#-tecnologías-y-herramientas)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Dataset](#-dataset)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Uso](#-uso)
- [Arquitecturas Implementadas](#-arquitecturas-implementadas)
- [Resultados](#-resultados)
- [Autor y Créditos](#-autor-y-créditos)
- [Licencia](#-licencia)

---

## 🎯 Descripción del Proyecto

El proyecto aborda el problema de **clasificación automática del grado de artrosis de rodilla** (según la escala Kellgren-Lawrence, KL 0-4) mediante el uso de **redes neuronales convolucionales profundas (CNNs)** y **Vision Transformers (ViT)**. 

### Objetivos principales:

- ✅ Comparar diferentes arquitecturas CNN (desde modelos pequeños hasta grandes redes preentrenadas)
- ✅ Evaluar enfoques de clasificación multiclase vs. binaria
- ✅ Analizar modelos de regresión vs. clasificación
- ✅ Implementar técnicas de validación cruzada con conjuntos externos (imágenes de control)
- ✅ Aplicar técnicas de interpretabilidad (Grad-CAM)

### Flujo de trabajo:

1. **Preprocesamiento y aumento de datos**: Uso de `ImageDataGenerator` (TensorFlow) y transformaciones en PyTorch
2. **Definición de modelos**: Implementación de arquitecturas CNN y ViT personalizadas y preentrenadas
3. **Entrenamiento**: Con callbacks de early stopping, reducción de learning rate y validación en conjunto externo
4. **Evaluación**: Métricas de accuracy, MAE, matrices de confusión y curvas de aprendizaje
5. **Interpretabilidad**: Mapas de activación Grad-CAM para visualizar zonas relevantes

---

## 🛠️ Tecnologías y Herramientas

### Frameworks de Deep Learning
- **TensorFlow/Keras** (v2.10.0) - Para modelos CNN
- **PyTorch** (v1.13.1) - Para Vision Transformers
- **PyTorch Lightning** (v1.8.6) - Para organización de experimentos

### Librerías principales
- **NumPy** - Operaciones numéricas
- **Matplotlib/Seaborn** - Visualización de resultados
- **OpenCV** - Procesamiento de imágenes
- **Weights & Biases (wandb)** - Tracking de experimentos
- **scikit-learn** - Métricas y evaluación

### Arquitecturas preentrenadas
- EfficientNet (B0, B4, B5, B7)
- ResNet (18, 50 V2, 151)
- Vision Transformer (ViT)

---

## 📁 Estructura del Repositorio

```
Radiography_TFG/
│
├── notebooks/                      # Todos los experimentos y análisis
│   ├── kaggle_experiments/         # Experimentos con CNNs en Kaggle
│   ├── vision_transformers/        # Experimentos con ViT en PyTorch
│   └── validation_experiments/     # Validación externa y Grad-CAM
│
├── documentos/                     # Documentación del TFG
│   ├── memoria_latex/              # Código LaTeX de la memoria
│   ├── papers/                     # Papers de referencia
│   └── pdf_checkpoints/            # Versiones de la memoria
│
├── requirements.txt                # Dependencias del proyecto
├── LICENSE                         # Licencia MIT
└── README.md                       # Este archivo
```

**Nota**: Los directorios `data/`, `models/` y `dataset/` están excluidos del repositorio (ver `.gitignore`). Descarga el dataset desde Kaggle (ver sección [Dataset](#-dataset)).

---

## 🗂️ Dataset

El proyecto utiliza el **Knee Osteoarthritis Dataset with Severity Grading** disponible en Kaggle:

🔗 [Knee X-ray Dataset - Kaggle](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

### Características del dataset:

- **Clases**: 5 grados de artrosis según escala Kellgren-Lawrence (0-4)
- **Splits**: Train, Validation, Test
- **Formato**: Imágenes de rayos X en escala de grises

### Cómo obtener el dataset:

1. Descarga el dataset desde Kaggle
2. Extrae los archivos en una carpeta `dataset/` en la raíz del proyecto
3. Asegúrate de que la estructura sea: `dataset/train/`, `dataset/val/`, `dataset/test/`

**Validación externa**: Se utilizó un conjunto adicional de imágenes de gatos para monitorizar el sobreajuste (validación cruzada).

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- CUDA (opcional, para entrenamiento en GPU)
- Cuenta de Weights & Biases (opcional, para tracking)

### Instalación

1. **Clonar el repositorio**:

```bash
git clone https://github.com/Tempus23/Radiography_TFG.git
cd Radiography_TFG
```

2. **Crear entorno virtual** (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

4. **Configurar Weights & Biases** (opcional):

```bash
wandb login
```

---

## 💻 Uso

### Estructura de los notebooks

Todos los notebooks siguen una estructura similar:

1. **Importación de librerías y utilidades**
2. **Carga y preparación de datos** (generadores de datos)
3. **Definición del modelo** (arquitectura CNN o ViT)
4. **Configuración de callbacks** (early stopping, reducción LR, validación externa)
5. **Entrenamiento del modelo**
6. **Evaluación y visualización** (métricas, matrices de confusión, curvas)

### Ejecutar experimentos

Los notebooks están diseñados para ejecutarse en **Google Colab** o **Kaggle Kernels**:

1. Abre el notebook deseado en la plataforma
2. Asegúrate de tener acceso al dataset (monta Google Drive o añade el dataset de Kaggle)
3. Ejecuta las celdas secuencialmente

**Notebooks destacados**:

- `notebooks/kaggle_experiments/A_BEST_ONE_EfficientNetB5_reg_origOAI.ipynb` - Mejor modelo de regresión
- `notebooks/vision_transformers/PytorchViT-T0.ipynb` - Vision Transformer base
- `notebooks/validation_experiments/EffB0-0-gradcam.ipynb` - Interpretabilidad con Grad-CAM

### Reproducir resultados

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar un notebook (ejemplo con Jupyter)
jupyter notebook notebooks/kaggle_experiments/A_BEST_ONE_EfficientNetB5_reg_origOAI.ipynb
```

**Importante**: Si no tienes el dataset de gatos para validación externa, elimina o comenta el callback correspondiente en los notebooks.

---

## 🏗️ Arquitecturas Implementadas

### Modelos CNN (TensorFlow/Keras)

| Modelo | Descripción | Parámetros |
|--------|-------------|------------|
| **SmallCNN** | CNN personalizada con 3 capas convolucionales | ~500K |
| **MidCNN** | CNN mediana con 5 capas convolucionales | ~2M |
| **BigCNN** | CNN profunda con 7+ capas convolucionales | ~5M |
| **EfficientNetB0-B7** | Arquitecturas EfficientNet preentrenadas | 5M - 66M |
| **ResNet18-151** | ResNet con diferentes profundidades | 11M - 60M |

### Modelos ViT (PyTorch)

| Modelo | Descripción | Parámetros |
|--------|-------------|------------|
| **PytorchViT** | Vision Transformer base | ~85M |
| **PytorchLargeViT** | Vision Transformer Large | ~300M+ |

### Enfoques de entrenamiento

- **Clasificación multiclase** (5 clases: KL 0-4)
- **Clasificación binaria** (sano vs. artrosis)
- **Regresión** (predicción continua del grado)
- **Transfer Learning** (fine-tuning de modelos preentrenados)
- **Ensemble** (combinación de múltiples modelos)

---

## 📊 Resultados

### Mejor modelo: EfficientNetB5 (Regresión)

- **MAE en test**: ~0.45
- **Accuracy (redondeado)**: ~75%
- **Tipo**: Transfer learning + fine-tuning
- **Dataset**: OAI (Osteoarthritis Initiative)

### Hallazgos clave

✅ Los modelos de regresión superan a los de clasificación en métricas generales  
✅ EfficientNet B5 ofrece el mejor balance entre rendimiento y eficiencia  
✅ La validación con imágenes externas (gatos) ayuda a detectar overfitting temprano  
✅ Grad-CAM muestra que los modelos se centran en zonas articulares relevantes  
✅ El aumento de datos mejora significativamente la generalización  

### Visualizaciones

Los notebooks incluyen:

- 📈 Curvas de aprendizaje (loss y métricas)
- 🔥 Mapas de calor (Grad-CAM)
- 🎯 Matrices de confusión
- 📊 Reportes de clasificación detallados

---

## 👤 Autor y Créditos

### Autor

**Carlos Hernández Martínez**

- 🎓 Estudiante de Ingeniería Informática
- 🏛️ Universidad Politécnica de Valencia (UPV)
- 🔗 GitHub: [@Tempus23](https://github.com/Tempus23)
- 📧 Contacto: A través de GitHub

### Colaboración

Este proyecto se desarrolla en colaboración con una **tesis veterinaria** en curso sobre diagnóstico de artrosis en animales.

### Agradecimientos

- Profesores y tutores del TFG
- Comunidad de Kaggle por el dataset
- Desarrolladores de TensorFlow, PyTorch y Weights & Biases

---

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia MIT** - ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 Carlos Hernández Martínez

Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia
de este software para usar, copiar, modificar, fusionar, publicar y distribuir
el software, sujeto a las condiciones de la licencia MIT.
```

---

## 📖 Cómo Citar

Si utilizas este proyecto en tu investigación o trabajo, por favor cita:

```
Hernández Martínez, C. (2025). Clasificación de Artrosis de Rodilla mediante 
Redes Convolucionales Profundas. Trabajo de Fin de Grado, Universidad Politécnica 
de Valencia. https://github.com/Tempus23/Radiography_TFG
```

---

## 📚 Referencias

- Kellgren, J. H., & Lawrence, J. S. (1957). Radiological assessment of osteo-arthrosis.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs.
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition.

---

## 🔗 Enlaces Útiles

- [Documentación TensorFlow](https://www.tensorflow.org/api_docs)
- [Documentación PyTorch](https://pytorch.org/docs/)
- [Weights & Biases](https://wandb.ai/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**

*Última actualización: Octubre 2025*
