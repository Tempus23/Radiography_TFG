# Función para entrenar el modelo
import wandb
import torch

def train_model(model, train_loader, val_loader, trainer, device, num_epochs=25, classification=True, learning_rate=0.001, betas=(0.9, 0.999), factor=0.1, patience=5):
    """
    Train the given model
    """
    model.to(device)
    optimizer, scheduler = trainer.configure_optimizers(learning_rate=learning_rate, betas=betas, factor=factor, patience=patience)

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
                ACC_value = res['ACC']
                recall_value = res['recall'].item()
                precision_value = res['precision'].item()
                f1_score_value = res['f1_score'].item()
                AUC_value = res['AUC']

                # Calcular promedios
                epoch_loss += loss_value
                avg_loss = epoch_loss / (iteration + 1)

                # Loggear métricas a la mitad
                if (iteration % len(loader) // 4 == 0) and phase == 'train':
                    if phase == 'train':
                        wandb.log({"train_loss": avg_loss, "train_acc": ACC_value,
                                    "train_recall": recall_value, "train_precision": precision_value,
                                    "train_f1_score": f1_score_value, "epoch": epoch, "progress" : iteration / len(loader), "train_AUC" : AUC_value})

                if phase == 'train':
                        wandb.log({"train_loss": avg_loss, "train_acc": ACC_value,
                                    "train_recall": recall_value, "train_precision": precision_value,
                                    "train_f1_score": f1_score_value, "epoch": epoch, "progress" : iteration / len(loader), "train_AUC" : AUC_value})
                else:
                    wandb.log({"val_loss": avg_loss, "val_acc": ACC_value,
                                "val_recall": recall_value, "val_precision": precision_value,
                                "val_f1_score": f1_score_value, "epoch": epoch,  "val_AUC" : AUC_value})
               
                    

            # Calcular métricas promedio por época
            trainer.restart_epoch(plot = False)

            # Ajustar el scheduler después de la fase de validación
            print(f" Epoch: {epoch}, Phase: {phase}, Loss: {avg_loss:.4f}, ACC: {ACC_value:.4f}, AUC: {AUC_value:.4f}", end = "")
            if phase == 'val':
                print()
                scheduler.step(avg_loss)

def test_model(model, test_loader, trainer, device, classification=True):
    """
    Test the given model
    """
    model.eval()
    model.to(device)

    epoch_loss = 0.0
    avg_loss = 0.0
    trainer.restart_epoch(plot = False)
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        
        labels = labels.to(device) if classification else labels.to(device).float()

        with torch.no_grad():
            res = trainer.validation_step(inputs, labels)
            # Extraer valores escalares
            loss = res['loss']
            
        loss_value = loss.item()
        # Calcular promedios
        epoch_loss += loss_value

    ACC_value = res['ACC']
    recall_value = res['recall'].item()
    precision_value = res['precision'].item()
    f1_score_value = res['f1_score'].item()
    AUC_value = res['AUC']
    avg_loss = epoch_loss / len(test_loader)

    wandb.log({"test_loss": avg_loss, "test_acc": ACC_value.item(),
                "test_recall": recall_value, "test_precision": precision_value,
                "test_f1_score": f1_score_value, "test_AUC" : AUC_value})
    
    print(f"Test model {model.__class__.__name__} - Loss: {avg_loss:.2f}, ACC: {ACC_value:.2f}, AUC: {AUC_value:.2f}")

    trainer.restart_epoch(plot = True)