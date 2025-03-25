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
    def __init__(self, model, device, L1=0.001, L2=0.001, lr=0.001, patience=5, factor=0.1, betas=(0.9, 0.999), num_classes=5):
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
        self.normalize = 4.0
        self.loss = nn.MSELoss()
        self.linearLoss = nn.L1Loss()
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        self.auc_metric = tm.AUROC(task="binary").to(device)
        
        # Initialize metrics logging
        self.train_metrics = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": [], "auc": []}
        self.val_metrics = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": [], "auc": []}

    def forward(self, x):
        return self.model(x)
        
    def prediction(self, y_hat):
        y_hat = y_hat * self.normalize
        # Improved clipping with cleaner torch operations
        y_hat = torch.clamp(y_hat, min=0.0, max=4.0)
        return torch.round(y_hat).float()

    def training_step(self, x, y):
        y = y / self.normalize

        y_hat = self.model(x)
        y = y.float()
        
        # Ensure dimensions match for loss calculation
        if y_hat.dim() > y.dim():
            y_hat = y_hat.squeeze()
        
        loss = self.loss(y_hat, y)
        
        # Regularization with more efficient computation
        if self.L1 > 0:
            L1_reg = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
        else:
            L1_reg = 0
        if self.L2 > 0:
            L2_reg = sum(torch.sum(param ** 2) for param in self.model.parameters())
        else:
            L2_reg = 0
        L1_reg = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
        L2_reg = sum(torch.sum(param ** 2) for param in self.model.parameters())
        
        # Separate loss components for better tracking
        prediction_loss = loss.detach()
        regularized_loss = loss + self.L1 * L1_reg + self.L2 * L2_reg
        
        # Transform back to original scale for metrics
        y_orig = y * self.normalize
        y_pred = self.prediction(y_hat)
        
        # Update metrics
        self.confusion_matrix.update(y_pred, y_orig.int())
        self.auc_metric.update(y_hat, y_orig.int())

        precision, recall, f1_score, ACC, AUC, specificity = self.calculate_metrics_from_confusion_matrix()
        
        # Log metrics
        self.log('train_loss', prediction_loss, prog_bar=True)
        self.log('train_acc', ACC, prog_bar=True)
        self.log('train_f1', f1_score)
        self.log('train_auc', AUC)

        return {"loss": regularized_loss, "base_loss": prediction_loss, "ACC": ACC, "recall": recall, 
                "precision": precision, "f1_score": f1_score, "AUC": AUC, "specificity": specificity}

    def validation_step(self, x, y):
        y = y / self.normalize
        y_hat = self.model(x)
        y = y.float()
        
        # Ensure dimensions match
        if y_hat.dim() > y.dim():
            y_hat = y_hat.squeeze()
            
        loss = self.loss(y_hat, y)
        linear_loss = self.linearLoss(y_hat, y)
        
        # Convert back for metrics
        y_orig = y * self.normalize
        y_pred = self.prediction(y_hat)
        
        self.confusion_matrix.update(y_pred, y_orig.int())
        self.auc_metric.update(y_hat, y_orig.int())

        precision, recall, f1_score, ACC, AUC, specificity = self.calculate_metrics_from_confusion_matrix()
        
        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', ACC, prog_bar=True)
        self.log('val_f1', f1_score)
        self.log('val_auc', AUC)
        
        return {"loss": loss, "linear_loss": linear_loss, "ACC": ACC, "precision": precision, 
                "recall": recall, "f1_score": f1_score, "AUC": AUC, "specificity": specificity}
                
    def on_train_epoch_end(self):
        self.restart_epoch(plot=False)
        
    def on_validation_epoch_end(self):
        self.restart_epoch(plot=False)

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
