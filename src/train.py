import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from wandb import wandb
import os

def create_tqdm_bar(iterable, desc, mode):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=200, desc=desc)

def train_model(model, trainer, train_dataset, val_dataset, epochs=5, transform=None, 
                device='cuda', save_model=False, name="Test", wdb=True, local=False, 
                project="oai-knee-cartilage-segmentation", early_stopping_patience=10, 
                plot_loss=True):
    if wdb:
        save_model = True
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=project,
            name=f"{model.__class__.__name__}_{name}",
            # track hyperparameters and run metadata
            config={
                "model": model.__class__.__name__,
                "Batch_size": train_dataset.batch_size,
                "learning_rate": trainer.learning_rate,
                "L1": trainer.L1,
                "L2": trainer.L2,
                "patience": trainer.patience,
                "factor": trainer.factor,
                "betas": trainer.betas,
                "epochs": epochs,
            }
        )
    train_loader = train_dataset.get_dataloader(shuffle=True)
    val_loader = val_dataset.get_dataloader(shuffle=True)
    model.to(device)
    history = train(model, train_loader, val_loader, trainer, epochs, device, wdb, local=local, 
                    save_model=save_model, early_stopping_patience=early_stopping_patience)
    
    # Plot training & validation loss if requested
    if plot_loss and history:
        plot_training_history(history, model.__class__.__name__, name, save_model)
    

def train(model, train_loader, val_loader, trainer, epochs, device, wdb, 
          local=False, save_model=True, early_stopping_patience=10):
    """
    train the given model and return training history
    """
    optimizer, scheduler = trainer.configure_optimizers()
    best_model = None
    best_loss = float('inf')
    best_epoch = -1
    early_stop_counter = 0
    
    # History dictionary to store metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epochs': []
    }
    
    for epoch in range(epochs):        
        training_loss = []
        validation_loss = []

        training_loss_num = 0
        complete_loss_num = 0
        validation_loss_num = 0

        # use training data
        model.train()

        training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{epochs}]', mode='train')
        for train_iteration, batch in training_loop:
            batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            res = trainer.training_step(batch[0], batch[1])
            optimizer.step()

            training_loss.append(res['loss'].item())
            training_loss_num += res['loss'].item()
            complete_loss_num += res['real_loss'].item()
            # Update the progress bar.
            training_loop.set_postfix(train_loss="{:.4f}".format(training_loss_num / (train_iteration + 1)),
                                      complete_loss="{:.4f}".format(complete_loss_num / (train_iteration + 1)),
                                      acc=res['ACC'].item(),
                                      AUC=res['AUC'].item(),
                                      sensivity=res['recall'].item(),
                                      specificity=res['specificity'].item())
            if wdb:
                
                wandb.log({"train_loss": training_loss_num / (train_iteration + 1),
                           "complete_loss": complete_loss_num / (train_iteration + 1),
                        "train_acc": res['ACC'],
                        "train_recall": res['recall'].item(),
                        "train_precision": res['precision'].item(),
                        "train_specifity": res['specificity'].item(),
                        "train_f1_score": res['f1_score'].item(),
                        "train_AUC": res['AUC'],
                        "epoch": epoch + 1,
                        "learning_rate": optimizer.param_groups[0]['lr']})
        
        # Store training metrics for this epoch
        avg_train_loss = training_loss_num / (train_iteration + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(res['ACC'].item())
        history['epochs'].append(epoch + 1)
        
        trainer.restart_epoch(plot=False)
        # use validation data
        if local:
            continue
            
        model.eval()
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]', mode='val')
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                batch = batch[0].to(device), batch[1].to(device)
                res = trainer.validation_step(batch[0], batch[1])  
                validation_loss.append(res['loss'].item())
                validation_loss_num += res['loss'].item()
                val_loop.set_postfix(val_loss="{:.8f}".format(validation_loss_num / (val_iteration + 1)),
                                     acc=res['ACC'].item(),
                                     AUC=res['AUC'].item(),
                                     specificity=res['specificity'].item())
        
        # Calculate average validation loss for this epoch
        avg_val_loss = validation_loss_num / (val_iteration + 1)
        
        # Store validation metrics for this epoch
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(res['ACC'].item())
                                      
        if wdb:
            wandb.log({"val_loss": avg_val_loss,
                    "val_acc": res['ACC'],
                    "val_recall": res['recall'].item(),
                    "val_precision": res['precision'].item(),
                    "val_specificity": res['specificity'].item(),
                    "val_f1_score": res['f1_score'].item(),
                    "val_AUC": res['AUC'],
                    "epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]['lr']})
                    
        if avg_val_loss < best_loss:
            early_stop_counter = 0
            best_epoch = epoch
            best_loss = avg_val_loss
            best_model = model
            if save_model != "":
                model_path = f"best_model_{model.__class__.__name__}_epoch_{epoch + 1}.pt"
                torch.save(model, model_path)
                if wdb:
                    wandb.save(model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                if wdb:
                    wandb.run.summary["stopped_early"] = True
                    wandb.run.summary["total_epoch_trained"] = epoch + 1
                break
        scheduler.step(avg_val_loss)
        trainer.restart_epoch(plot=False)
    
    # Mark the best epoch in history
    history['best_epoch'] = best_epoch + 1
    history['best_val_loss'] = best_loss
    
    test_model(best_model, val_loader, trainer, device, wdb)
    
    return history

def test_model(model, test_loader, trainer, device, wdb=False):
    """
    Test the given model
    """
    model.eval()
    model.to(device)

    epoch_loss = 0.0
    avg_loss = 0.0
    trainer.restart_epoch(plot=False)
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        
        labels = labels.to(device)

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
    if wdb:
        wandb.log({"test_loss": avg_loss, "test_acc": ACC_value.item(),
                "test_recall": recall_value, "test_precision": precision_value,
                "test_f1_score": f1_score_value, "test_AUC" : AUC_value})
    
    print(f"Test model {model.__class__.__name__} - Loss: {avg_loss:.2f}, ACC: {ACC_value:.2f}, AUC: {AUC_value:.2f}, Sensivility: {recall_value:.2f}, Specificity: {precision_value:.2f}")

    trainer.restart_epoch(plot=True)

def plot_training_history(history, model_name, experiment_name, save_model=None):
    """
    Plot the training and validation loss/accuracy history
    
    Args:
        history (dict): Dictionary with training history data
        model_name (str): Name of the model
        experiment_name (str): Name of the experiment
        save_model (str): If provided, save the plot to a file with this prefix
    """
    plt.figure(figsize=(15, 6))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
    
    # Mark best epoch
    if 'best_epoch' in history:
        best_idx = history['epochs'].index(history['best_epoch'])
        plt.axvline(x=history['best_epoch'], color='green', linestyle='--', 
                   label=f'Best Model (Epoch {history["best_epoch"]})')
        plt.plot(history['best_epoch'], history['val_loss'][best_idx], 'go', 
                markersize=10, label=f'Best Val Loss: {history["best_val_loss"]:.4f}')
    
    plt.title(f'Loss Evolution - {model_name} {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy')
    
    # Mark best epoch
    if 'best_epoch' in history:
        best_idx = history['epochs'].index(history['best_epoch'])
        plt.axvline(x=history['best_epoch'], color='green', linestyle='--', 
                   label=f'Best Model (Epoch {history["best_epoch"]})')
        plt.plot(history['best_epoch'], history['val_acc'][best_idx], 'go', 
                markersize=10, label=f'Val Acc: {history["val_acc"][best_idx]:.4f}')
    
    plt.title(f'Accuracy Evolution - {model_name} {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.show()