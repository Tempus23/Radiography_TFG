# Función para entrenar el modelo
import wandb
import torch

def train_model(model, train_loader, val_loader, trainer, optimizer, device, num_epochs=25, classification=True):
    """
    Train the given model
    """
    model.to(device)
    optimizer = trainer.configure_optimizers()
    scheduler = trainer.configure_scheduler(optimizer)

    for epoch in range(num_epochs):


        # Training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            epoch_loss = 0.0
            avg_loss = 0.0


            for iteration, batch in enumerate(loader):
                # Iterar sobre los batches
                inputs, labels = batch
                inputs = inputs.to(device)

                labels = labels.to(device) if classification else labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        res = trainer.training_step(inputs, labels)
                        loss = res['loss']
                        optimizer.step()
                    else:
                        with torch.no_grad():
                            res = trainer.validation_step(inputs, labels)
                            loss = res['loss']

                # Extraer valores escalares
                loss_value = loss.item()
                accuracy_value = res['accuracy'].item()
                recall_value = res['recall'].item()
                precision_value = res['precision'].item()
                f1_score_value = res['f1_score'].item()

                # Calcular promedios
                epoch_loss += loss_value
                avg_loss = epoch_loss / (iteration + 1)

    
                if phase == 'train':
                    wandb.log({"train_loss": avg_loss, "train_acc": accuracy_value,
                                "train_recall": recall_value, "train_precision": precision_value,
                                "train_f1_score": f1_score_value, "epoch": epoch, "progress" : iteration / len(loader)})
                else:
                    wandb.log({"val_loss": avg_loss, "val_acc": accuracy_value,
                                "val_recall": recall_value, "val_precision": precision_value,
                                "val_f1_score": f1_score_value, "epoch": epoch})

            # Calcular métricas promedio por época
            if phase == 'val':
                trainer.restart_epoch(plot = True)
            else:
                trainer.restart_epoch(plot = False)

            # Ajustar el scheduler después de la fase de validación
            print(f" Epoch: {epoch}, Phase: {phase},  Loss: {avg_loss:.2f}, Accuracy: {accuracy_value:.2f}, Recall: {recall_value:.2f}, Precision: {precision_value:.2f}, F1 Score: {f1_score_value:.2f}")
            if phase == 'val':
               
                scheduler.step(avg_loss)

def test_model(model, test_loader, trainer, device, classification=True):
    """
    Test the given model
    """
    model.eval()
    model.to(device)

    epoch_loss = 0.0
    avg_loss = 0.0

    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        
        labels = labels.to(device) if classification else labels.to(device).float()

        with torch.no_grad():
            res = trainer.validation_step(inputs, labels)
            loss = res['loss']

            # Calcular promedios
            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(test_loader)
    precision_value, recall_value, f1_score_value, accuracy_value = trainer.calculate_metrics_from_confusion_matrix()

    wandb.log({"test_loss": avg_loss, "test_acc": accuracy_value.item(),
                "test_recall": recall_value.item(), "test_precision": precision_value.item(),
                "test_f1_score": f1_score_value.item()})
    
    print(f"Test model {model.__class__.__name__} - Loss: {avg_loss:.2f}, Accuracy: {accuracy_value.item():.2f}, Recall: {recall_value.item():.2f}, Precision: {precision_value.item():.2f}, F1 Score: {f1_score_value.item():.2f}")

    trainer.restart_epoch(plot = True)