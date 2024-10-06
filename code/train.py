import torch
from tqdm import tqdm

from definitions import BATCH_SIZE
from data import contrast_transform

def create_tqdm_bar(iterable, desc, mode):
    return tqdm(enumerate(iterable),total=len(iterable), ncols=150, desc=desc)
# Función para entrenar el modelo
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        #Training phase
        model.train()
        train_loss_gen = 0.0
        train_corrects = 0

        training_loop = create_tqdm_bar(train_loader, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]', mode='train')
        for train_iteration, batch in training_loop:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_gen += loss.item() * BATCH_SIZE
            train_corrects += torch.sum(preds == labels.data).double()

            train_loss = train_loss_gen / ((train_iteration + 1)*BATCH_SIZE)
            train_acc = train_corrects / ((train_iteration + 1)*BATCH_SIZE)
            
            training_loop.set_postfix(train_loss = "{:.8f}".format(train_loss), train_acc = "{:.8f}".format(train_acc))

        #Validation phase
        model.eval()
        val_loss_gen = 0.0
        val_corrects = 0
        
        validation_loop = create_tqdm_bar(val_loader, desc=f'Validation Epoch [{epoch + 1}/{num_epochs}]', mode='val')
        with torch.no_grad():
            for val_iteration, batch in validation_loop:
                
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss_gen += loss.item() * BATCH_SIZE
                val_corrects += torch.sum(preds == labels.data).double()
                
                val_loss = val_loss_gen / ((val_iteration + 1)*BATCH_SIZE)
                val_acc = val_corrects / ((val_iteration + 1)*BATCH_SIZE)
                validation_loop.set_postfix(val_loss = "{:.8f}".format(val_loss), val_acc = "{:.8f}".format(val_acc ))

    return model