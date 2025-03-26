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
        y = y.float() / self.normalize

        y_hat = self.model(x)
        
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
        
        regularized_loss.backward()
        # Transform back to original scale for metrics
        y_orig = y * self.normalize
        y_pred = self.prediction(y_hat)
        
        # Update metrics
        self.confusion_matrix.update(y_pred, y_orig.int())
        self.auc_metric.update(y_hat, y_orig.int())

        precision, recall, f1_score, ACC, AUC, specificity = self.calculate_metrics_from_confusion_matrix()
        
        # Log metrics

        return {"loss": prediction_loss, "real_loss": regularized_loss, "ACC": ACC, "recall": recall, 
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

        
    def plot_predictions_on_line(self, dataloader, epoch=0, max_samples=2000):
        """
        Visualiza las estadísticas de predicciones del modelo en una recta numérica,
        mostrando la media y los rangos de confianza (75% y 95%) para cada clase.

        Args:
            dataloader: Dataloader con los datos a evaluar.
            epoch: Número de época actual para el título.
            max_samples: Máximo número de muestras a considerar.
        """
        self.model.eval()
        predictions_by_class = {}
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                # Manejar el batch según el formato (tuple/list) o solo datos
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                else:
                    x, y = batch, None

                device = next(self.model.parameters()).device
                x = x.to(device)
                if y is not None:
                    y = y.to(device)

                y_hat = self.model(x)
                if y_hat.dim() > 1:
                    y_hat = y_hat.squeeze()

                # Convertir predicciones al rango original
                y_hat_scaled = y_hat * self.normalize

                # Convertir tensores a arrays de NumPy
                y_cpu = y.cpu().numpy() if y is not None else np.zeros_like(y_hat_scaled.cpu().numpy())
                y_hat_cpu = y_hat_scaled.cpu().numpy()

                # Agrupar predicciones por clase
                for cls in np.unique(y_cpu):
                    cls_int = int(cls)
                    mask = y_cpu == cls
                    if cls_int not in predictions_by_class:
                        predictions_by_class[cls_int] = []
                    predictions_by_class[cls_int].extend(y_hat_cpu[mask].tolist())

                count += len(y)
                if count >= max_samples:
                    break

        # Configurar figura y ejes usando subplots
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Usar colormap para asignar colores automáticamente
        cmap = plt.get_cmap('tab10')
        classes = sorted(predictions_by_class.keys())
        offsets = np.linspace(-0.3, 0.3, len(classes))
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
        
        stats_table = []
        headers = ["Clase", "Media", "Mediana", "P25", "P75", "P5", "P95", "MAE", "Muestras"]
        
        scatter_handles = []  # Para las medias de cada clase
        for i, cls in enumerate(classes):
            preds = np.array(predictions_by_class[cls])
            mean = np.mean(preds)
            median = np.median(preds)
            p25 = np.percentile(preds, 25)
            p75 = np.percentile(preds, 75)
            p5 = np.percentile(preds, 5)
            p95 = np.percentile(preds, 95)
            mae = np.mean(np.abs(preds - cls))
            
            stats_table.append([str(cls), f"{mean:.2f}", f"{median:.2f}", 
                                f"{p25:.2f}", f"{p75:.2f}", f"{p5:.2f}", f"{p95:.2f}",
                                f"{mae:.3f}", str(len(preds))])
            
            color = cmap(i % 10)
            
            # Línea vertical para la posición real de la clase
            ax.axvline(x=cls, color='gray', linestyle='--', alpha=0.5)
            
            # Graficar la media como punto
            size = min(300, max(100, len(preds) / 5))
            sc = ax.scatter(mean, offsets[i], color=color, s=size, edgecolors='black', zorder=10)
            scatter_handles.append(sc)
            
            # Graficar rangos: línea gruesa para 25%-75% y línea fina para 5%-95%
            ax.plot([p25, p75], [offsets[i], offsets[i]], color=color, linewidth=4, alpha=0.7)
            ax.plot([p5, p95], [offsets[i], offsets[i]], color=color, linewidth=2, alpha=0.4)
        
        ax.set_xlim(-0.5, self.num_classes - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(f'Estadísticas de predicciones por clase (Época {epoch})')
        ax.set_xlabel('Predicción del modelo')
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks([])  # Ocultar eje Y
        
        # Añadir etiquetas superiores para cada clase
        for i in range(self.num_classes):
            ax.text(i, 0.45, f'Clase {i}', ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))
        
        # Leyenda para las medias de las clases
        legend1 = ax.legend(scatter_handles, [f'Clase {cls}' for cls in classes],
                            title='Clases', loc='upper center',
                            bbox_to_anchor=(0.5, -0.15), ncol=min(3, len(classes)))
        ax.add_artist(legend1)
        
        # Leyenda adicional para los rangos de confianza
        custom_lines = [
            plt.Line2D([0], [0], color='gray', lw=4, alpha=0.7),
            plt.Line2D([0], [0], color='gray', lw=2, alpha=0.4)
        ]
        ax.legend(custom_lines, ['Rango 25%-75%', 'Rango 5%-95%'],
                loc='upper right', framealpha=0.9)
        
        fig.tight_layout()
        plt.show()
        
        # Mostrar la tabla de estadísticas en consola
        print("\nEstadísticas detalladas por clase:")
        try:
            from tabulate import tabulate
            print(tabulate(stats_table, headers=headers, tablefmt="grid"))
        except ImportError:
            print(" | ".join(headers))
            print("-" * 80)
            for row in stats_table:
                print(" | ".join(row))
        
        # Visualizar la tabla en una figura de matplotlib
        fig_table, ax_table = plt.subplots(figsize=(12, len(classes)*0.8 + 1.5))
        ax_table.axis('off')
        table = ax_table.table(cellText=stats_table, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title("Estadísticas de predicciones por clase")
        fig_table.tight_layout()
        plt.show()

