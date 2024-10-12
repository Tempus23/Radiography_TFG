import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt

class Classification(pl.LightningModule):
    def __init__(self, model, device):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.model.classes).to(device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # Obtener la clase predicha
        y_pred = torch.argmax(y_hat, dim=1)
        # Calcular métricas
        loss.backward()
        self.confusion_matrix.update(y_pred, y)

        precision, recall, f1_score, accuracy = self.calculate_metrics_from_confusion_matrix()

        return {"loss": loss, "accuracy": accuracy, "recall": recall, "precision": precision, "f1_score": f1_score}

    def validation_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # Obtener la clase predicha
        y_pred = torch.argmax(y_hat, dim=1)
        # Calcular métricas
        self.confusion_matrix.update(y_pred, y)

        precision, recall, f1_score, accuracy = self.calculate_metrics_from_confusion_matrix()

        return {"loss": loss, "accuracy": accuracy, "precision" : precision, "recall": recall, "f1_score" : f1_score}


    def restart_epoch(self):
        self.confusion_matrix.plot()
        plt.show()
        self.confusion_matrix.reset()
        self.accuracy.reset()
        self.recall.reset()
        self.precision.reset()
        self.f1_score.reset()
        self.confusion_matrix.reset()
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

      # Calcular Accuracy
      accuracy = true_positives.sum() / confusion_matrix.sum()
      # Retornar las métricas
      return precision, recall, f1, accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999))

    def configure_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                          factor=0.1,
                                                          patience=5)

class Regression(pl.LightningModule):
    def __init__(self, model, device):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.loss = nn.MSELoss()
        self.linearLoss = nn.L1Loss()
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.model.classes).to(device)

    def forward(self, x):
        return self.model(x)
    def prediction(self, y_hat):
        # Comparaciones y operaciones con tensores en PyTorch
        y_hat = torch.where(y_hat < 0.5, torch.tensor(0.0), y_hat)
        y_hat = torch.where(y_hat > 3.5, torch.tensor(4.0), y_hat)
        return torch.round(y_hat).long()

    def training_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss(y_hat.squeeze(), y)
        loss.backward()
        linear_loss = self.linearLoss(y_hat.squeeze(), y)

        y_pred = self.prediction(y_hat)
        # Calcular el número de aciertos
        self.confusion_matrix.update(y_pred.squeeze(), y)

        precision, recall, f1_score, accuracy = self.calculate_metrics_from_confusion_matrix()

        return {"loss": linear_loss, "accuracy": accuracy, "precision" : precision, "recall": recall, "f1_score" : f1_score}

    def validation_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss(y_hat.squeeze(), y)
        linear_loss = self.linearLoss(y_hat.squeeze(), y)
        #Redondear y_hat para obtener la clase predicha
        y_hat_rounded = self.prediction(y_hat)

         # Calcular el número de aciertos
        y_pred = self.prediction(y_hat)
        self.confusion_matrix.update(y_pred.squeeze(), y)

        precision, recall, f1_score, accuracy = self.calculate_metrics_from_confusion_matrix()

        return {"loss": linear_loss, "accuracy": accuracy, "precision" : precision, "recall": recall, "f1_score" : f1_score}
    def restart_epoch(self):
        self.confusion_matrix.plot()
        plt.show()
        self.confusion_matrix.reset()

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

      # Calcular Accuracy
      accuracy = true_positives.sum() / confusion_matrix.sum()
      # Retornar las métricas
      return precision, recall, f1, accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999))

    def configure_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.1,
                                                           patience=5)


