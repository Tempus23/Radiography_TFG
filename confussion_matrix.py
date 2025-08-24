import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# ==============================
# 🔹 1. Matriz de confusión fija
# ==============================
cm = np.array([
    [20, 0, 2, 0, 0],
    [27, 0, 3, 0, 0],
    [14, 0, 4, 0, 0],
    [9,  0, 2, 0, 0],
    [2,  0, 0, 0, 0]
])

labels = [0, 1, 2, 3, 4]

# ==============================
# 🔹 2. Dibujar matriz
# ==============================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Etiqueta verdadera")
plt.title("Matriz de confusión (datos fijos)")
plt.show()

# ==============================
# 🔹 3. Calcular métricas básicas
# ==============================
# Reconstruimos y_true / y_pred a partir de la matriz
y_true, y_pred = [], []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        y_true.extend([i]*cm[i,j])
        y_pred.extend([j]*cm[i,j])

print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nReporte por clase:\n", classification_report(y_true, y_pred, labels=labels, zero_division=0))
