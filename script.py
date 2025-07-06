import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Estilo estético para gráficos
sns.set(style="whitegrid")

# Datos de precisión multiclase (KL 0–4)
data = {
    "Modelo": [
        "CNN simple",
        "CNN mediana",
        "CNN profunda",
        "EfficientNetB0",
    ],
    "Precisión (%)": [
        42.00,
        46.85,
        58.47,
        68.64,
    ]
}

df = pd.DataFrame(data)

# Crear la figura
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=df, x="Modelo", y="Precisión (%)", palette="Blues_d")

# Anotar cada barra con el valor numérico
for p in barplot.patches:
    height = p.get_height()
    barplot.annotate(f'{height:.2f}%',
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='bottom',
                     fontsize=10)

# Títulos y ajustes
plt.title("Precisión multiclase (KL 0-4) por modelo", fontsize=14)
plt.ylim(0, 80)
plt.ylabel("Precisión (%)")
plt.xlabel("Modelo")
plt.tight_layout()

# Guardar como PDF
plt.savefig("accuracy_modelos.pdf")
plt.show()
