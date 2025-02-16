import torch
from tqdm import tqdm
from wandb import wandb

def create_tqdm_bar(iterable, desc, mode):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=100, desc=desc)

def train_model(model, trainer, train_dataset, val_dataset, epochs=5, transform=None, device='cuda', name="Test"):
    wandb.init(
        project="oai-knee-cartilage-segmentation",
        name = name,

        # track hyperparameters and run metadata
        config={
            "model": model.name,
            "Batch_size" : train_dataset.batch_size,
            "learning_rate": trainer.learning_rate,
            "L1": trainer.L1,
            "L2": trainer.L2,
            "patience": trainer.patience,
            "factor": trainer.factor,
            "betas": trainer.betas,
            "epochs": epochs,
        }
    )
    train_loader = train_dataset.get_dataloader()
    val_loader = val_dataset.get_dataloader()
    model.to(device)
    train(model, train_loader, val_loader, trainer, epochs, device)
    




def train(model, train_loader, val_loader, trainer, epochs, device):
    """
    train the given model
    """
    optimizer, scheduler = trainer.configure_optimizers()
    
    for epoch in range(epochs):        
        training_loss = []
        validation_loss = []

        training_loss_num = 0
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
            # Update the progress bar.
            training_loop.set_postfix(curr_train_loss = "{:.8f}".format(training_loss_num / (train_iteration + 1)), val_loss = "{:.8f}".format(validation_loss_num))
            wandb.log({"train_loss": training_loss_num / (train_iteration + 1),
                        "train_acc": res['ACC'],
                        "train_recall": res['recall'].item(),
                        "train_precision": res['precision'].item(),
                        "train_f1_score": res['f1_score'].item(),
                        "train_AUC": res['AUC'],
                        "epoch": epoch})
        
        # use validation data
        model.eval()
        val_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{epochs}]', mode='val')
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                batch = batch[0].to(device), batch[1].to(device)
                res = trainer.validation_step(batch[0], batch[1])  
                validation_loss.append(res['loss'].item())
                validation_loss_num += res['loss'].item()
                val_loop.set_postfix(val_loss = "{:.8f}".format(validation_loss_num / (val_iteration + 1)))
                wandb.log({"val_loss": validation_loss_num / len(val_loop),
                            "val_acc": res['ACC'],
                            "val_recall": res['recall'].item(),
                            "val_precision": res['precision'].item(),
                            "val_f1_score": res['f1_score'].item(),
                            "val_AUC": res['AUC'],
                            "epoch": epoch})
        scheduler.step(res['loss'].item())