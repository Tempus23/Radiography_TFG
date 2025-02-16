import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt


class Classification(pl.LightningModule):
    """
    Trainer para entrenar un modelo de clasificación multiclase
    y de dimension 1 con valores [0, num_classes]
    """
    def __init__(self, model, device, L1=0.001, L2=0.001, lr=0.001, patience=5, factor=0.1, betas=(0.9, 0.999)):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()
        self.L1 = L1
        self.L2 = L2
        self.learning_rate = lr
        self.patience = patience
        self.factor = factor
        self.betas = betas


        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=5).to(device)
        self.auc_metric = tm.AUROC(num_classes=5, task="multiclass").to(device)  # Definir métrica AUROC para clasificación multiclase

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y = y
        y_hat = self.model(x)
        y_oh = self.transform_classes(y)
        loss = self.loss_fn(y_hat, y_oh)
        
        # Regularización L1
        L1_reg = torch.tensor(0., requires_grad=True)
        for param in self.model.parameters():
            L1_reg = L1_reg + torch.sum(torch.abs(param))
        
        # Regularización L2
        L2_reg = torch.tensor(0., requires_grad=True)
        for param in self.model.parameters():
            L2_reg = L2_reg + torch.sum(param ** 2)
        
        # Añadir regularización a la pérdida
        loss = loss + self.L1 * L1_reg + self.L2 * L2_reg
        # Obtener la clase predicha
        y_pred = torch.argmax(y_hat, dim=1)
        # Calcular métricas
        loss.backward()
        self.confusion_matrix.update(y_pred, y)
        self.auc_metric.update(y_hat, y)

        precision, recall, f1_score, ACC, AUC = self.calculate_metrics_from_confusion_matrix()

        return {"loss": loss, "ACC": ACC, "recall": recall, "precision": precision, "f1_score": f1_score, "AUC": AUC}

    def validation_step(self, x, y):
        y = y
        y_hat = self.model(x)
        y_oh = self.transform_classes(y)
        loss = self.loss_fn(y_hat, y_oh)
        # Obtener la clase predicha
        y_pred = torch.argmax(y_hat, dim=1)
        # Calcular métricas
        self.confusion_matrix.update(y_pred, y)
        self.auc_metric.update(y_hat, y)

        precision, recall, f1_score, ACC, AUC = self.calculate_metrics_from_confusion_matrix()

        return {"loss": loss, "ACC": ACC, "precision" : precision, "recall": recall, "f1_score" : f1_score, "AUC": AUC}

    def transform_classes(self, y):
        # Convertir las clases a un formato de one-hot encoding
        return torch.nn.functional.one_hot(y.to(torch.int64), num_classes=5).to(float).squeeze()
    def restart_epoch(self, plot = False):
        if plot:
            self.confusion_matrix.plot()
            plt.show()
        self.confusion_matrix.reset()
        self.auc_metric.reset()
    def calculate_metrics_from_confusion_matrix(self):
      confusion_matrix = self.confusion_matrix.compute()
      # Verdaderos positivos por clase (diagonal de la matriz)
      true_positives = torch.diag(confusion_matrix)

      # Predicciones totales por clase (sumar columnas)
      predicted_positives = confusion_matrix.sum(dim=0)

      # Ejemplos reales por clase (sumar filas)
      actual_positives = confusion_matrix.sum(dim=1)

      # Calcular Precision, Recall, F1 por clase
      precision = (true_positives / (predicted_positives + 1e-8)).mean()  # Añadir pequeña constante para evitar división por 0
      recall = (true_positives / (actual_positives + 1e-8)).mean()
      f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

      # Calcular ACC
      ACC = true_positives.sum() / confusion_matrix.sum()

      # Calcular el AUC
      AUC = self.auc_metric.compute()
      # Retornar las métricas
      return precision, recall, f1, ACC, AUC

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     betas=self.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               factor=self.factor,
                                                               patience=self.patience)
        return optimizer, scheduler
