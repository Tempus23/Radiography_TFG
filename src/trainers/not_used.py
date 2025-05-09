import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt

class LabelClassification(pl.LightningModule):
    """
    Trainer para entrenar un modelo de clasificación multilabel
    y de dimension N con valores [0, 1]*N
    """
    def __init__(self, model, device):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.model.classes).to(device)
        self.auc_metric = tm.AUROC(num_classes=self.model.classes, task="multiclass").to(device)  # Definir métrica AUROC para clasificación multiclase

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y = y.squeeze()
        y_hat = self.model(x)
        y_oh = self.transform_classes(y)
        loss = self.loss_fn(y_hat, y_oh)
        # Obtener la clase predicha
        y_pred = torch.argmax(y_hat, dim=1)
        # Calcular métricas
        loss.backward()
        self.confusion_matrix.update(y_pred, y)
        self.auc_metric.update(y_hat, y)

        precision, recall, f1_score, ACC, AUC = self.calculate_metrics_from_confusion_matrix()

        return {"loss": loss, "ACC": ACC, "recall": recall, "precision": precision, "f1_score": f1_score, "AUC": AUC}

    def validation_step(self, x, y):
        y = y.squeeze()
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
        return torch.nn.functional.one_hot(y.to(torch.int64), num_classes=self.model.classes).to(float).squeeze()
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

    def configure_optimizers(self, learning_rate=0.001, betas=(0.9, 0.999), factor=0.1, patience=5):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=learning_rate,
                                     betas=betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               factor=factor,
                                                               patience=patience)
        return optimizer, scheduler



class BinaryClassification(pl.LightningModule):
    """
    Trainer para entrenar un modelo de clasificación binaria
    y de dimension 1 con valores [0, 1]
    """
    def __init__(self, model, device):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.confusion_matrix = tm.ConfusionMatrix(num_classes=model.classes, task="binary").to(device)
        self.auc_metric = tm.AUROC(task="binary").to(device)  # Definir métrica AUROC para clasificación binaria

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y.float())
        # Obtener la clase predicha
        y_pred =y_hat >= 0.5
        # Calcular métricas
        loss.backward()
        self.confusion_matrix.update(y_pred.int(), y.int())
        self.auc_metric.update(torch.sigmoid(y_hat), y.int())


        precision, recall, f1_score, ACC, auc = self.calculate_metrics_from_confusion_matrix()

        return {"loss": loss, "ACC": ACC, "recall": recall, "precision": precision, "f1_score": f1_score, "AUC": auc}

    def validation_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y.float())
        # Obtener la clase predicha
        y_pred = y_hat >= 0.5
        # Calcular métricas
        self.confusion_matrix.update(y_pred.int(), y.int())
        self.auc_metric.update(torch.sigmoid(y_hat), y.int())


        precision, recall, f1_score, ACC, auc = self.calculate_metrics_from_confusion_matrix()

        return {"loss": loss, "ACC": ACC, "precision" : precision, "recall": recall, "f1_score" : f1_score, "AUC": auc}

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
        return precision, recall, f1, ACC, AUC
    
    def configure_optimizers(self, learning_rate=0.001, betas=(0.9, 0.999), factor=0.1, patience=5):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=learning_rate,
                                     betas=betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               factor=factor,
                                                               patience=patience)
        return optimizer, scheduler


