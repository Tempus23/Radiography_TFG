import torch
from tqdm import tqdm
from wandb import wandb

def create_tqdm_bar(iterable, desc, mode):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=200, desc=desc)

def train_model(model, trainer, train_dataset, val_dataset, epochs=5, transform=None, 
                device='cuda', save_model = False, name="Test", wdb=True, local=False, 
                project="oai-knee-cartilage-segmentation", early_stopping_patience=10):
    if wdb:
        save_model = True
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=project,
            name=name,
            # track hyperparameters and run metadata
            config={
                "model": model.name,
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
    train(model, train_loader, val_loader, trainer, epochs, device, wdb, local = local, 
          save_model = save_model, early_stopping_patience=early_stopping_patience)
    

def train(model, train_loader, val_loader, trainer, epochs, device, wdb, 
          local = False, save_model = True, early_stopping_patience=10):
    """
    train the given model
    """
    optimizer, scheduler = trainer.configure_optimizers()
    best_model = None
    best_loss = float('inf')
    best_epoch = -1
    early_stop_counter = 0
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
                val_loop.set_postfix(val_loss = "{:.8f}".format(validation_loss_num / (val_iteration + 1)),
                                      acc=res['ACC'].item(),
                                      AUC=res['AUC'].item(),
                                      specificity=res['specificity'].item())
                                      
        if wdb:
            wandb.log({"val_loss": validation_loss_num / (val_iteration + 1),
                    "val_acc": res['ACC'],
                    "val_recall": res['recall'].item(),
                    "val_precision": res['precision'].item(),
                    "val_specificity": res['specificity'].item(),
                    "val_f1_score": res['f1_score'].item(),
                    "val_AUC": res['AUC'],
                    "epoch": epoch + 1,
                    "learning_rate": optimizer.param_groups[0]['lr']})
        if validation_loss_num < best_loss:
            early_stop_counter = 0
            best_epoch = epoch
            best_loss = validation_loss_num
            if save_model != "":
                model_path = f"best_model_{model.__class__.__name__}_{save_model}_epoch_{epoch + 1}.pt"
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
        scheduler.step(validation_loss_num)
        trainer.restart_epoch(plot=False)
    
    test_model(model, val_loader, trainer, device, wdb)

def test_model(model, test_loader, trainer, device, wdb=False):
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

    trainer.restart_epoch(plot = True)