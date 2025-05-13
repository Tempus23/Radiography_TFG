# IA_KneeXRay

# TFG: Clasificación de Artrosis de Rodilla mediante Redes Convolucionales Profundas

Este proyecto forma parte del Trabajo de Fin de Grado (TFG) en Ingeniería Informática y tiene como objetivo el desarrollo, entrenamiento y evaluación de modelos de deep learning para la clasificación automática del grado de artrosis de rodilla a partir de radiografías.

## Descripción general

Se utilizan diferentes arquitecturas de redes neuronales convolucionales (CNN), desde modelos sencillos hasta arquitecturas profundas, para abordar la tarea de clasificación multiclase y binaria del grado de artrosis (escala KL 0-4). El flujo de trabajo incluye:

- **Preprocesamiento y generación de datos:**
  - Uso de `ImageDataGenerator` para aumentar y normalizar imágenes.
  - Generación de dataframes a partir de la estructura de carpetas del dataset.
- **Definición de modelos:**
  - Implementación de varias arquitecturas CNN (pequeña, mediana, grande).
  - Posibilidad de ajustar para tareas de regresión o clasificación.
- **Entrenamiento y validación:**
  - Entrenamiento con callbacks personalizados, validación en un conjunto especial de imágenes de gatos para monitorizar el sobreajuste.
  - Uso de técnicas como EarlyStopping y ReduceLROnPlateau.
- **Evaluación:**
  - Cálculo de métricas como accuracy, MAE, matriz de confusión y reporte de clasificación.
  - Visualización de curvas de aprendizaje y resultados.

## Estructura de los notebooks

1. **Importación de librerías y utilidades**
2. **Generación de generadores de datos y dataframes**
3. **Definición de callbacks personalizados para validación**
4. **Definición de arquitecturas CNN**
5. **Entrenamiento del modelo**
6. **Evaluación y visualización de resultados**

## Dataset

Se emplea un dataset de kaggle de radiografías de rodilla etiquetadas según el grado de artrosis (escala KL). El dataset está dividido en particiones de entrenamiento, validación, test y un conjunto especial de imágenes de gatos para validación cruzada.

## Objetivo

El objetivo es comparar el rendimiento de diferentes arquitecturas y enfoques (clasificación vs regresión, multiclase vs binaria) para determinar la mejor estrategia de diagnóstico automático de artrosis de rodilla.

## Autor

Este proyecto ha sido desarrollado por Carlos Hernández Martínezcomo parte del TFG en la Universidad Politécnica de Valéncia y como colaboración de una tesis Veterinaria en curso.

---

**Nota:** Para reproducir los experimentos, consulta los requisitos en `requirements.txt` y asegúrate de tener los datasets organizados según la estructura indicada en el notebook. En caso de no tener el dataset de gatos u otro de validación externa quita el callback utilizado para ello
