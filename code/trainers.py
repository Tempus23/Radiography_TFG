import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics as tm
import torch.autograd as autograd


class Classification(pl.LightningModule):
    def __init__(self, model, n_classes: int):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.loss = nn.CrossEntropyLoss()

        metrics = tm.MetricCollection(
            {
                "top1": tm.Accuracy(task="multiclass", top_k=1, num_classes=n_classes),
                "top3": tm.Accuracy(task="multiclass", top_k=3, num_classes=n_classes),
                "avg-prec": tm.AveragePrecision(task="multiclass", num_classes=n_classes),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def classify(self, x):
        logits = self.model(x)
        return nn.functional.softmax(logits, dim=-1)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.model(x)
        loss = self.loss(logits, labels)

        self.log("train/loss", loss.mean(), batch_size=x.shape[0])
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        self.log_dict(
            self.val_metrics(self.classify(x), labels),
            prog_bar=True,
            batch_size=x.shape[0],
        )

    def test_step(self, batch, batch_idx):
        x, labels = batch
        self.log_dict(
            self.test_metrics(self.classify(x), labels),
            prog_bar=True,
            batch_size=x.shape[0],
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

class Regression(pl.LightningModule):
    def __init__(self, model, device):
        super().__init__()
        self.save_hyperparameters(ignore=("model",))

        self.model = model
        self.loss = nn.MSELoss()
        self.linearLoss = nn.L1Loss()
        self.accuracy = tm.Accuracy(task="multiclass", num_classes=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss(y_hat.squeeze(), y)
        loss.backward()
        linear_loss = self.linearLoss(y_hat.squeeze(), y)
        
        #Determinar si el modelo acierta la clase correcta [0, 1, 2, 3, 4]
        #Redondear y_hat para obtener la clase predicha
        y_hat_rounded = torch.round(y_hat).squeeze()
        # Calcular el número de aciertos
        correct = torch.sum(y_hat_rounded == y).item()

        return {"loss": linear_loss, "corrects": correct}

    def validation_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss(y_hat.squeeze(), y)
        linear_loss = self.linearLoss(y_hat.squeeze(), y)
        #Redondear y_hat para obtener la clase predicha
        y_hat_rounded = torch.round(y_hat).squeeze()

         # Calcular el número de aciertos
        correct = torch.sum(y_hat_rounded == y).item()

        return {"loss": linear_loss, "corrects": correct}


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999))
    
    def configure_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                           factor=0.1, 
                                                           patience=5)
