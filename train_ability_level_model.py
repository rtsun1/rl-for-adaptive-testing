import dataloader as dl
from torch.utils.data import Dataset, DataLoader
from ability_level_model import UNet

import time
import tqdm
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_



def split_into_n_questions(X, num_question_range, device):
    like_X = torch.ones_like(X) * -1
    Xs = random.sample(range(X.shape[1]), random.randint(num_question_range[0], num_question_range[1]))
    like_X[Xs] = X[Xs]
    like_X = like_X.to(device)
    return like_X


def loss_fn(pred, target):
    return torch.nn.functional.mse_loss(pred, target)

def validation(model, validation_loader, num_question_range, device):
    avg_loss = 0.
    pbar_val = tqdm.tqdm(validation_loader)
    for sample in pbar_val:
        with torch.no_grad():
            x, label = sample[0].to(device), sample[1].to(device)
            x = split_into_n_questions(x, num_question_range, device)
            estimate = model(x)
            loss = loss_fn(estimate, label)
            avg_loss += loss.item()
    
    return avg_loss / len(validation_loader)


def train_ability_level_model(train_loader: DataLoader, 
                              validation_loader: DataLoader,
                              epochs: int,
                              num_question_range = [1, 20], 
                              lr: float=2e-4, 
                              device=torch.device('cuda'), 
                              eval_intervals: int=10):
    torch.cuda.empty_cache()

    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    
    pbar_epochs = tqdm.tqdm(range(epochs))
    for epoch in pbar_epochs:
        avg_loss = 0.
        start = time.time()
        pbar_samples = tqdm.tqdm(train_loader, leave=False)
        for sample in pbar_samples:
            x, label = sample[0].to(device), sample[1].to(device)
            x = split_into_n_questions(x, num_question_range, device)
            estimate = model(x)
            
            optimizer.zero_grad()
            loss = loss_fn(estimate, label)
            
            loss.backward()
            
            try:
                clip_grad_norm_(model.parameters(), 1.)
            except:
                pass
            
            optimizer.step()

            avg_loss += loss.item()

        avg_loss /= len(train_loader)
        
        # validating and saving the model
        print(f'Epoch: {epoch} - Loss: {avg_loss:.3f} ({time.time() - start:.2f} sec)')
        lr_scheduler.step(avg_loss)

        if epoch % eval_intervals == 0 or epoch == epochs - 1:
            # generate some sample to see the quality of the generative model
            avg_val_loss = validation(model, validation_loader, num_question_range, device)
            
            torch.save(model.state_dict(), f'checkpoints/val_loss_{round(avg_val_loss, 5)}.pt')


if __name__ == "__main__":
    EPOCHS = 200
    BATCH_SIZE = 128
    NUM_QUESTION_RANGE = [1, 20]

    train, val, test = dl.dataloader_ml(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE)
    train_ability_level_model(train, val, EPOCHS, NUM_QUESTION_RANGE)
