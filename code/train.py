# Función para entrenar el modelo
import wandb
import torch

def train_model(model, train_loader, val_loader, trainer, optimizer, device, num_epochs=25):
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
            epoch_accuracy = 0.0
            epoch_recall = 0.0
            epoch_precision = 0.0
            epoch_f1_score = 0.0
            num_batches = 0
            avg_loss = 0.0
            avg_accuracy = 0.0
            avg_recall = 0.0
            avg_precision = 0.0
            avg_f1_score = 0.0


            for iteration, batch in enumerate(loader):
                # Iterar sobre los batches
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device).float()

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

                # Acumular métricas
                epoch_loss += loss_value
                epoch_accuracy += accuracy_value
                epoch_recall += recall_value
                epoch_precision += precision_value
                epoch_f1_score += f1_score_value
                num_batches += 1

                # Media de las metricas
                avg_loss = epoch_loss / (iteration + 1)
                avg_accuracy = epoch_accuracy / (iteration + 1)
                avg_recall = epoch_recall / (iteration + 1)
                avg_precision = epoch_precision / (iteration + 1)
                avg_f1_score = epoch_f1_score / (iteration + 1)

                # Actualizar la barra de progreso

                # Registrar métricas en wandb
                if phase == 'train':
                    wandb.log({"train_loss": avg_loss, "train_acc": avg_accuracy,
                                "train_recall": avg_recall, "train_precision": avg_precision,
                                "train_f1_score": avg_f1_score, "epoch": epoch})
                else:
                    wandb.log({"val_loss": avg_loss, "val_acc": avg_accuracy,
                                "val_recall": avg_recall, "val_precision": avg_precision,
                                "val_f1_score": avg_f1_score, "epoch": epoch})
            print(avg_accuracy)

            # Calcular métricas promedio por época
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            avg_recall = epoch_recall / num_batches
            avg_precision = epoch_precision / num_batches
            avg_f1_score = epoch_f1_score / num_batches

            print(f'{phase.capitalize()} Loss: {avg_loss:.4f} Acc: {avg_accuracy:.4f} '
                  f'Recall: {avg_recall:.4f} Precision: {avg_precision:.4f} F1: {avg_f1_score:.4f}')
            trainer.restart_epoch()

            # Ajustar el scheduler después de la fase de validación
            if phase == 'val':
                scheduler.step(avg_loss)
