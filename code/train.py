import torch
from tqdm import tqdm

from definitions import BATCH_SIZE

def create_tqdm_bar(iterable, desc, mode):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)
# Función para entrenar el modelo
def train_model(model, train_loader, val_loader, trainer, optimizer, device, num_epochs=25):
    """
    Train the given model
    """
    optimizer = trainer.configure_optimizers()
    scheduler = trainer.configure_scheduler(optimizer)


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        #Training phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            train_loss = 0.0
            train_corrects = 0
            val_loss = 0.0
            val_corrects = 0
            if phase == 'train':
                training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]', mode='train')
            else:
                training_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{num_epochs}]', mode='val')
            for iteration, batch in training_loop:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    if phase == 'train':
                        res = trainer.training_step(inputs, labels)
                        optimizer.step()
                    else:
                        res = trainer.validation_step(inputs, labels)

                    loss = res['loss']
                    n_corrects = res['corrects']
                        
                if phase == 'train':
                    train_loss += loss.item()
                    train_corrects += n_corrects
                else:
                    val_loss += loss.item()
                    val_corrects += n_corrects

                if phase == 'train':
                    epoch_loss = train_loss / (iteration + 1)
                    epoch_acc = train_corrects / ((iteration + 1)*BATCH_SIZE)
                    training_loop.set_postfix(curr_train_loss = "{:.4f}".format(epoch_loss), curr_train_acc = "{:.4f}".format(epoch_acc))
                else:
                    epoch_loss = val_loss / (iteration + 1)
                    epoch_acc = val_corrects / ((iteration + 1)*BATCH_SIZE)
                    training_loop.set_postfix(curr_val_loss = "{:.4f}".format(epoch_loss), curr_val_acc = "{:.4f}".format(epoch_acc))
                
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        scheduler.step(val_loss)    