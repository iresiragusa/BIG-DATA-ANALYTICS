from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import torch

class EarlyStopping:
    # Stops training when validation loss does not improve for `patience` consecutive epochs.
    # The best model weights are saved to disk and can be restored at the end of training.

    def __init__(self, patience = 10, min_delta = 1e-4,
                 checkpoint_path = 'best_model.pt'):
        
        # patience        -- how many epochs to wait after the last improvement
        # min_delta       -- minimum change that qualifies as an improvement
        # checkpoint_path -- file where the best state_dict is saved
        
        self.patience        = patience
        self.min_delta       = min_delta
        self.checkpoint_path = checkpoint_path

        self.best_loss   = float('inf')
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module):
        # Call once per epoch with the current validation loss.
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found: reset counter and save checkpoint
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def train_one_epoch(model, loader, criterion, optimizer, device):
    # Run one full pass over the training set and return the average loss.
    model.train()   # activates Dropout layers
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()                           # reset gradients from previous step
        logits = model(X_batch)                         # forward pass
        loss   = criterion(logits, y_batch)             # compute cross-entropy
        loss.backward()                                 # backpropagation
        optimizer.step()                                # update weights

        total_loss += loss.item()                       # accumulate weighted loss

    return total_loss / len(loader.dataset)

def evaluate_epoch(model, loader, criterion, device):
    # Evaluate the model on a DataLoader; return average loss and accuracy.
    with torch.no_grad():
        model.eval()    # deactivates Dropout layers
        total_loss, correct = 0.0, 0

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits, preds = model.predict(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item()
            # Predicted class = index with the highest logit
            preds   = preds.argmax(dim=1)
            correct += (preds == y_batch).sum().item()

        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct   / len(loader.dataset)
        return avg_loss, accuracy
    
def training(model, train_loader, val_loader, criterion, optimizer, early_stopping, device, max_epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in tqdm(range(1, MAX_EPOCHS + 1)):
        train_loss          = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc   = evaluate_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}')

        # Check early stopping after each epoch
        early_stopping.step(val_loss, model)
        if early_stopping.should_stop:
            print(f'\nEarly stopping triggered at epoch {epoch}.\nBest val loss: {early_stopping.best_loss:.4f}')
            break

    print('\nTraining complete.')

    return history, epoch

def print_curves(history):
    epochs_ran = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_ran, history['train_loss'], label='Train loss')
    ax1.plot(epochs_ran, history['val_loss'],   label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss curves')
    ax1.legend()

    ax2.plot(epochs_ran, history['val_acc'], color='green', label='Val accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def save_checkpoint(epoch, model, optimizer, val_loss, path):
        torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss'            : val_loss,
    }, path)
        
def load_model(path, model, optimizer):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    val_loss = checkpoint['val_loss']

    return model, optimizer, start_epoch, val_loss
        
def predict(model, loader, device):
    # Run inference and returns predicted class indices and class probabilities and golden labels

    with torch.no_grad():
        model.eval()   
        correct = 0

        y_preds = []
        y_score = []
        y_test = []

        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits, preds = model.predict(X_batch)

            log_logits = nn.LogSoftmax(dim=1)(logits)

            # Predicted class = index with the highest predicted score
            preds   = preds.argmax(dim=1)
            correct += (preds == y_batch).sum().item()

            y_preds.extend(preds)
            y_score.extend(log_logits)
            y_test.extend(y_batch)

        return y_test, y_preds, torch.stack(y_score)
        
