import dataloader as dl
from torch.utils.data import Dataset, DataLoader
from ability_level_model import RegressionModel

import time
import tqdm
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import pandas as pd
import plotly.figure_factory as ff
import numpy as np
from scipy.stats import norm

# Define constants for integration
QUADRATURE_POINTS = 41
ABILITY_RANGE = (-4, 4)

def calculate_eap_ability(responses, discrimination, difficulty):
    # Ensure everything is already on the same device (GPU or CPU)
    device = responses.device

    # Define quadrature points and weights for numerical integration
    abilities = torch.linspace(ABILITY_RANGE[0], ABILITY_RANGE[1], QUADRATURE_POINTS, device=device) 

    # Broadcast the abilities array to allow vectorized computation
    abilities_broadcasted = abilities.view(1, 1, -1)  # Shape: (1, 1, QUADRATURE_POINTS)
    
    # Compute the probability of correct response for all samples and items simultaneously
    prob_correct = 1 / (1 + torch.exp(-discrimination.unsqueeze(2) * (abilities_broadcasted - difficulty.unsqueeze(2))))

    # Compute the probability of incorrect response
    prob_incorrect = 1 - prob_correct

    # Compute the log likelihoods for correct and incorrect responses
    log_likelihood = responses.unsqueeze(2) * torch.log(prob_correct) + (1 - responses.unsqueeze(2)) * torch.log(prob_incorrect)

    # Sum log likelihoods over all items for each sample
    posterior_numerator = log_likelihood.sum(dim=1)  # Shape: (sample_size, QUADRATURE_POINTS)

    # Calculate posterior denominator with stability adjustment
    max_numerator = torch.max(posterior_numerator, dim=1, keepdim=True)[0]  # Stability trick to avoid overflow
    posterior_denominator = torch.exp(posterior_numerator - max_numerator)
    posterior_denominator /= posterior_denominator.sum(dim=1, keepdim=True)

    # Compute EAP ability for each sample
    eap_ability = torch.sum(abilities * posterior_denominator, dim=1)

    # Compute the standard error of the ability estimate for each sample
    eap_variance = torch.sum((abilities - eap_ability.unsqueeze(1)) ** 2 * posterior_denominator, dim=1)
    eap_standard_error = torch.sqrt(eap_variance)

    return eap_ability, eap_standard_error

def split_into_n_questions(X, num_question_range, device):
    like_X = torch.ones_like(X) * -1
    Xs = random.sample(range(X.shape[1]), random.randint(num_question_range[0], num_question_range[1]))
    like_X[:, Xs, :] = X[:, Xs, :]
    like_X = like_X.to(device)
    return like_X, X[:, Xs, :]


def loss_fn(x, pred, target, x_full):
    #eap_full, eap_se_full = calculate_eap_ability(x_full[:, :, 2], x_full[:, :, 0], x_full[:, :, 1])
    pred_ability = torch.reshape(pred, target.shape)
    
    # loss for being outside of confidence interval of 0.01 * variation away from the true ability
    #CI_loss = (target - pred_ability).abs() - (1 * (eap_se_full ** 2))
    #CI_loss[CI_loss < 0] = 0
    #CI_loss = CI_loss ** 2
    #CI_loss = CI_loss.mean()

    true_loss = torch.nn.functional.mse_loss(pred_ability, target)
    return true_loss #CI_loss #+ 0.6 * true_loss

def loss_fn_validation(x, pred, target, loss_type="MSE"):
    eap, eap_se = calculate_eap_ability(x[:, :, 2], x[:, :, 0], x[:, :, 1])
    eap = torch.reshape(eap.float(), target.shape)

    if loss_type == "MSE":
        true_loss = torch.nn.functional.mse_loss(pred, target)
        loss_true = torch.nn.functional.mse_loss(eap, target)
    elif loss_type == "MAE":
        true_loss = torch.nn.functional.l1_loss(pred, target)
        loss_true = torch.nn.functional.l1_loss(eap, target)
    return true_loss, loss_true

def validation(model, validation_loader, num_question_range, device, validation_type="MAE"):
    avg_loss = 0.
    avg_loss_true = 0.
    pbar_val = tqdm.tqdm(validation_loader)
    for sample in pbar_val:
        with torch.no_grad():
            x, label = sample[0].to(device), sample[1].to(device).float()
            x, x_clean = split_into_n_questions(x, [15, 15], device)
            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            label = label[indices]
            estimate = model(x)
            label = torch.reshape(label, estimate.shape)
            loss, loss_true = loss_fn_validation(x_clean, estimate, label, validation_type)
            avg_loss += loss.item()
            avg_loss_true += loss_true.item()

    return avg_loss / len(validation_loader), avg_loss_true / len(validation_loader)


def train_ability_level_model(train_loader: DataLoader, 
                              validation_loader: DataLoader,
                              epochs: int,
                              num_question_range = [1, 20], 
                              model_name = None,
                              lr: float=2e-5, 
                              device=torch.device('cuda'), 
                              eval_intervals: int=30) -> None:
    torch.cuda.empty_cache()

    print("Loading model...")
    model = RegressionModel().to(device)
    if model_name is not None:
        model.load_state_dict(torch.load(model_name, weights_only=True))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    best_val_loss = -np.inf
    
    print("Training model...")
    for epoch in range(epochs):
        avg_loss = 0.
        start = time.time()
        pbar_samples = tqdm.tqdm(train_loader, leave=False)
        for sample in pbar_samples:
            x_full, label = sample[0].to(device), sample[1].to(device).float()
            x, clean_x = split_into_n_questions(x_full, num_question_range, device)
            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            label = label[indices]

            estimate = model(x)
            label = torch.reshape(label, estimate.shape)

            optimizer.zero_grad()
            loss = loss_fn(clean_x, estimate, label, x_full)
            loss.backward()
            
            optimizer.step()

            avg_loss += loss.item()
            pbar_samples.set_description(f"Loss: {loss.item():.3f}, Avg Loss: {avg_loss / (pbar_samples.n + 1):.3f}")
            #if pbar_samples.n % 5000 == 0:
            #   lr_scheduler.step(avg_loss/(pbar_samples.n + 1))

        avg_loss /= len(train_loader)
        #lr_scheduler.step(avg_loss)
        
        # validating and saving the model
        print(f'Epoch: {epoch} - Loss: {avg_loss:.3f} ({time.time() - start:.2f} sec)')
        

        if epoch % eval_intervals == 0 or epoch == epochs - 1:
            # generate some sample to see the quality of the generative model
            avg_val_loss, avg_true_loss = validation(model, validation_loader, num_question_range, device)
            print(f'Validation Loss: {avg_val_loss:.3f}, Scoring Loss: {avg_true_loss:.3f}')
            if (avg_true_loss - avg_val_loss) > best_val_loss:
                best_val_loss = avg_true_loss - avg_val_loss
                torch.save(model.state_dict(), f'checkpoints_AL_model/val_loss_best.pt')
        
        failure = True
        while failure:
            try:
                torch.save(model.state_dict(), f'checkpoints_AL_model/last_epoch.pt')
                failure = False
            except:
                print("Failed to save model, retrying...")
                time.sleep(0.05)
            
            

def loss_fn_test(x, pred, target, loss_type="MSE"):
    pred_ability = pred
    eap, eap_se = calculate_eap_ability(x[:, :, 2], x[:, :, 0], x[:, :, 1])
    eap = torch.reshape(eap.float(), target.shape)

    TEST_ESTIMATES[0].append(pred_ability.cpu().detach().numpy()[0])
    TEST_ESTIMATES[1].append(eap.cpu().detach().numpy()[0])
    TEST_ESTIMATES[2].append(target.cpu().detach().numpy()[0])
    weight_func = (20 + target.shape[0]) / 90 if target.shape[0] < 70 else 1
    weight_model = 1 - weight_func
    TEST_ESTIMATES[3].append((weight_model * pred_ability + weight_func * eap).cpu().detach().numpy()[0])

    if loss_type == "MSE":
        pred_loss = torch.nn.functional.mse_loss(pred_ability, target)
        loss_true = torch.nn.functional.mse_loss(eap, target)
        avg_loss = torch.nn.functional.mse_loss((pred_ability + eap) / 2, target)
        ratio_loss = torch.nn.functional.mse_loss((weight_model * pred_ability + weight_func * eap), target)
    elif loss_type == "MAE":
        pred_loss = torch.nn.functional.l1_loss(pred_ability, target)
        loss_true = torch.nn.functional.l1_loss(eap, target)
        avg_loss = torch.nn.functional.l1_loss((pred_ability + eap) / 2, target)
        ratio_loss = torch.nn.functional.l1_loss((weight_model * pred_ability + weight_func * eap), target)
    elif loss_type == "RMSE":
        pred_loss = torch.sqrt(torch.nn.functional.mse_loss(pred_ability, target))
        loss_true = torch.sqrt(torch.nn.functional.mse_loss(eap, target))
        avg_loss = torch.sqrt(torch.nn.functional.mse_loss((pred_ability + eap) / 2, target))
        ratio_loss = torch.sqrt(torch.nn.functional.mse_loss((weight_model * pred_ability + weight_func * eap), target))
    
    TEST_LOSSES[0].append(pred_loss.cpu().detach().numpy())
    TEST_LOSSES[1].append(loss_true.cpu().detach().numpy())
    TEST_LOSSES[2].append(ratio_loss.cpu().detach().numpy())
    return pred_loss, loss_true

def test_func(model, validation_loader, num_question_range, device, validation_type="MSE"):
    avg_loss = 0.
    avg_loss_true = 0.
    for sample in validation_loader:
        with torch.no_grad():
            x, label = sample[0].to(device), sample[1].to(device).float()
            x, x_clean = split_into_n_questions(x, num_question_range, device)
            estimate = model(x)
            label = torch.reshape(label, estimate.shape)
            loss, loss_true = loss_fn_test(x_clean, estimate, label, validation_type)
            avg_loss += loss.item()
            avg_loss_true += loss_true.item()

    return avg_loss / len(validation_loader), avg_loss_true / len(validation_loader)


TEST_ESTIMATES = [[], [], [], []]
TEST_LOSSES = [[], [], []]
if __name__ == "__main__":
    EPOCHS = 10000
    BATCH_SIZE = 8#1024#128#32#2048
    NUM_QUESTION_RANGE = [1, 15]

    print("Loading data...")
    train, val, test = dl.dataloader_ml(BATCH_SIZE, 1, 1)

    #train_ability_level_model(train, val, EPOCHS, NUM_QUESTION_RANGE, eval_intervals=1)
    train_ability_level_model(train, val, EPOCHS, NUM_QUESTION_RANGE, "checkpoints_AL_model/last_epoch.pt", eval_intervals=1)
    #train_ability_level_model(train, val, EPOCHS, NUM_QUESTION_RANGE, "checkpoints_AL_model/val_loss_best.pt", eval_intervals=1)

    """
    model = RegressionModel().to(torch.device('cuda'))
    model.load_state_dict(torch.load("checkpoints_AL_model/val_loss_best.pt", weights_only=True))

    pbar_test = tqdm.tqdm(range(1, 31))
    for i in pbar_test:
        TEST_ESTIMATES = [[], [], [], []]
        TEST_LOSSES = [[], [], []]
        avg_test_loss = 0.
        avg_true_func_loss = 0.
        pbar_epoch = tqdm.tqdm(range(1))
        for j in pbar_epoch:
            val_loss, true_loss = test_func(model, test, [i, i], torch.device('cuda'), "RMSE")
            avg_test_loss += val_loss
            avg_true_func_loss += true_loss
        avg_test_loss /= j+1
        avg_true_func_loss /= j+1

        mean_test_losses = np.mean(TEST_LOSSES, axis=1)
        TEST_ESTIMATES = np.array(TEST_ESTIMATES).reshape(4, -1)
        #print()
        #print(TEST_LOSSES)
        fig_estimates = ff.create_distplot(TEST_ESTIMATES, ["Predicted Model", "Function Provided", "Target", "Weighted Ratio"], bin_size=0.1)
        fig_estimates.update_layout(title=f"Estimates for {i} questions")
        fig_estimates.write_html(f"graphs/estimates_{i}.html")

        fig_losses = ff.create_distplot(TEST_LOSSES, ["Predicted Model -> " + str(round(mean_test_losses[0], 3)), "Function Provided -> " + str(round(mean_test_losses[1], 3)), "Weighted Ratio -> " + str(round(mean_test_losses[2], 3))], bin_size=0.1)
        fig_losses.update_layout(title=f"RMSE Losses for {i} questions")
        fig_losses.write_html(f"graphs/losses_{i}.html")
        #"""
