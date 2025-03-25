import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Regression(pl.LightningModule):
    """
    Trainer para entrenar un modelo de regresion ordinal
    y de 1dim con valores [0 - num_classes] con orden
    """
    def __init__(self, model, device, L1=0.001, L2=0.001, lr=0.001, patience=5, factor=0.1, betas=(0.9, 0.999), num_classes = 5):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.L1 = L1
        self.L2 = L2
        self.learning_rate = lr
        self.patience = patience
        self.factor = factor
        self.betas = betas

        self.model = model
        self.loss = nn.MSELoss()
        self.linearLoss = nn.L1Loss()
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        self.auc_metric = tm.AUROC(task="binary").to(device)  # Definir métrica AUROC para clasificación binaria

    def forward(self, x):
        return self.model(x)
    def prediction(self, y_hat):
        # Comparaciones y operaciones con tensores en PyTorch
        y_hat = torch.where(y_hat < 0.5, torch.tensor(0.0), y_hat)
        y_hat = torch.where(y_hat > 3.5, torch.tensor(4.0), y_hat)
        return torch.round(y_hat).float()

    def training_step(self, x, y):
        # Normalizar la x (max(x) = 5)
        x = x / 5.0

        y_hat = self.model(x)
        y = y.float()
        loss = self.loss(y_hat.squeeze(), y)
        # Regularización L1
        L1_reg = torch.tensor(0., requires_grad=True)
        for param in self.model.parameters():
            L1_reg = L1_reg + torch.sum(torch.abs(param))
        # Regularización L2
        L2_reg = torch.tensor(0., requires_grad=True)
        for param in self.model.parameters():
            L2_reg = L2_reg + torch.sum(param ** 2)
        
        # Añadir regularización a la pérdida
        prediction_loss = loss
        loss = loss + self.L1 * L1_reg + self.L2 * L2_reg
        # Calcular métricas
        loss.backward()

        y = y.float()
        y_pred = self.prediction(y_hat)
        # Calcular el número de aciertos
        self.confusion_matrix.update(y_pred.squeeze(), y.int())
        self.auc_metric.update(y_hat, y.int())

        precision, recall, f1_score, ACC, AUC, specificity = self.calculate_metrics_from_confusion_matrix()

        return {"loss": prediction_loss, "real_loss": loss, "ACC": ACC, "recall": recall, "precision": precision, "f1_score": f1_score, "AUC": AUC, "specificity": specificity}

    def validation_step(self, x, y):
        x = x / 5.0
        y_hat = self.model(x)
        y = y.float()
        loss = self.loss(y_hat, y)
        linear_loss = self.linearLoss(y_hat, y)
        #Redondear y_hat para obtener la clase predicha
        # Calcular el número de aciertos
        y_pred = self.prediction(y_hat)
        self.confusion_matrix.update(y_pred.squeeze(), y.int())
        self.auc_metric.update(y_hat, y.int())

        precision, recall, f1_score, ACC, AUC, specificity = self.calculate_metrics_from_confusion_matrix()
        return {"loss": loss, "ACC": ACC, "precision" : precision, "recall": recall, "f1_score" : f1_score, "AUC": AUC, "specificity": specificity}
    def restart_epoch(self, plot = False):
        if plot:
            self.plot()
        self.confusion_matrix.reset()
        self.auc_metric.reset()

    def calculate_metrics_from_confusion_matrix(self):
        # Obtener la matriz de confusión (suponiendo que es un tensor de torch)
        cm = self.confusion_matrix.compute()
        total_samples = cm.sum()
        
        # Verdaderos positivos por clase (diagonal de la matriz)
        true_positives = torch.diag(cm)
        
        # Predicciones totales por clase (sumar columnas)
        predicted_positives = cm.sum(dim=0)
        
        # Ejemplos reales por clase (sumar filas)
        actual_positives = cm.sum(dim=1)
        
        # Calcular falsos positivos y falsos negativos por clase
        false_positives = predicted_positives - true_positives
        false_negatives = actual_positives - true_positives
        
        # Calcular verdaderos negativos por clase
        true_negatives = total_samples - (actual_positives + predicted_positives - true_positives)
        
        # Calcular métricas por clase con robustez para evitar división por cero
        precision_per_class = true_positives / (predicted_positives + 1e-8)
        recall_per_class = true_positives / (actual_positives + 1e-8)
        specificity_per_class = true_negatives / (true_negatives + false_positives + 1e-8)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-8)
        
        # Promediar las métricas por clase
        precision = precision_per_class.mean()
        recall = recall_per_class.mean()
        specificity = specificity_per_class.mean()
        f1 = f1_per_class.mean()
        
        # Calcular Accuracy (exactitud)
        ACC = true_positives.sum() / total_samples
        
        # Calcular el AUC (suponiendo que self.auc_metric ya está correctamente definido)
        AUC = self.auc_metric.compute()
        
        return precision, recall, f1, ACC, AUC, specificity

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     betas=self.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               factor=self.factor,
                                                               patience=self.patience)
        return optimizer, scheduler

    def plot(self, epoch=0):
        # Computa la matriz de confusión y las métricas por clase
        cm = self.confusion_matrix.compute().cpu().numpy()
        support = cm.sum(axis=1)
        precision_per_class = np.diag(cm) / (cm.sum(axis=0) + 1e-8)
        recall_per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-8)
        
        # Cálculo de especificidad por clase
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TN = cm.sum() - (FP + FN + np.diag(cm))
        specificity_per_class = TN / (TN + FP + 1e-8)
        
        accuracy = np.diag(cm).sum() / cm.sum()

        # Crea dos subplots: uno para la matriz de confusión y otro para la tabla de métricas
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: Matriz de confusión con heatmap
        sns.heatmap(cm, annot=True, fmt="d", ax=axs[0], cmap="Blues")
        axs[0].set_title("Matriz de Confusión epoch " + str(epoch))
        axs[0].set_xlabel("Predicción")
        axs[0].set_ylabel("Real")

        # Subplot 2: Tabla de métricas por clase
        table_data = []
        for i in range(self.num_classes):
            table_data.append([f"Clase {i}",
                            f"{precision_per_class[i]:.2f}",
                            f"{recall_per_class[i]:.2f}",
                            f"{f1_per_class[i]:.2f}",
                            f"{specificity_per_class[i]:.2f}",
                            int(support[i])])
        axs[1].axis('tight')
        axs[1].axis('off')
        table = axs[1].table(cellText=table_data,
                            colLabels=["Clase", "Precision", "Sensivity/Recall", "F1", "Specificity", "Support"],
                            cellLoc="center", loc="center")
        axs[1].set_title(f"Metrics por clase\nAccuracy General: {accuracy:.2f}", pad=20)
