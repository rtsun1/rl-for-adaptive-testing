import dataloader as dl
from torch.utils.data import DataLoader
from ability_level_model import RegressionModel

import time
import tqdm
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, wilcoxon, ttest_ind
from statsmodels.stats.multitest import multipletests

from dataclasses import dataclass
import os

@dataclass
class Args:
    cuda: bool = True
    """Whether to use GPU or not"""
    lr: float = 2e-5
    """Learning rate for the model"""
    epochs: int = 10_000
    """Number of epochs to train the model"""
    training_batch_size: int = 8
    """Batch size for training"""
    question_range: tuple = (1, 244)
    """Range of number of questions to sample for each student"""
    validation_question_range: tuple = (15, 15)
    """Range of number of questions to sample for validation"""
    eval_intervals: int = 1
    """Number of epochs between evaluations"""
    dataset: str = "train"
    """Dataset to use [train, val, test]"""
    model_path: str|None = None
    """Path of the model weights to load [last, best, path_to_weights.pt]"""
    mode: str = "train"
    """Mode to run [train, test]"""
    train_seed: int = 0
    """Seed for training"""
    test_loss: str = "RMSE"
    """Loss function to use for testing [MSE, MAE, RMSE]"""
    test_epochs: int = 3
    """Number of epochs to test the model"""
    test_question_range: tuple = (1, 30)
    """Performance graphs for number of questions in this range"""
    verbose: bool = True
    """Whether to print verbose output"""


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
            x, x_clean = split_into_n_questions(x, num_question_range, device)
            indices = np.random.permutation(x.shape[0])
            x = x[indices]
            label = label[indices]
            estimate = model(x)
            label = torch.reshape(label, estimate.shape)
            loss, loss_true = loss_fn_validation(x_clean, estimate, label, validation_type)
            avg_loss += loss.item()
            avg_loss_true += loss_true.item()

    return avg_loss / len(validation_loader), avg_loss_true / len(validation_loader)


def train_ability_level_model(args: Args,
                              train_loader: DataLoader, 
                              validation_loader: DataLoader,
                              epochs: int,
                              num_question_range = [1, 20], 
                              model_name = None,
                              lr: float=2e-5, 
                              device=torch.device('cuda'), 
                              eval_intervals: int=30) -> None:
    torch.cuda.empty_cache()
    seed = args.train_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Loading model...")
    model = RegressionModel().to(device)
    if model_name is not None:
        model.load_state_dict(torch.load(model_name, weights_only=True))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    best_val_loss = -np.inf
    
    print("Training model...")
    train_loader.shuffle = True
    for epoch in range(epochs):
        avg_loss = 0.0
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
            avg_val_loss, avg_true_loss = validation(model, validation_loader, args.validation_question_range, device)
            if (avg_true_loss - avg_val_loss) > best_val_loss:
                best_val_loss = avg_true_loss - avg_val_loss
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(model.state_dict(), f'checkpoints/val_loss_best.pt')
            print(f'Validation Loss: {avg_val_loss:.3f}, Scoring Loss: {avg_true_loss:.3f}, Best Difference: {best_val_loss:.3f}')
        
        failure = True
        while failure:
            try:
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save(model.state_dict(), f'checkpoints/last_epoch.pt')
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

    if loss_type == "MSE":
        pred_loss = torch.nn.functional.mse_loss(pred_ability, target)
        loss_true = torch.nn.functional.mse_loss(eap, target)
    elif loss_type == "MAE":
        pred_loss = torch.nn.functional.l1_loss(pred_ability, target)
        loss_true = torch.nn.functional.l1_loss(eap, target)
    elif loss_type == "RMSE":
        pred_loss = torch.sqrt(torch.nn.functional.mse_loss(pred_ability, target))
        loss_true = torch.sqrt(torch.nn.functional.mse_loss(eap, target))
    
    TEST_LOSSES[0].append(pred_loss.cpu().detach().numpy())
    TEST_LOSSES[1].append(loss_true.cpu().detach().numpy())
    return pred_loss, loss_true

def test_func(args: Args, model, validation_loader, num_question_range, device, validation_type="MSE"):
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
    import tyro
    args = tyro.cli(Args)

    print("Loading data...")
    train, val, test = dl.dataloader_ml(args.training_batch_size, 1, 1)

    if args.dataset == "train":
        data = train
    elif args.dataset == "val":
        data = val
    elif args.dataset == "test":
        data = test
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("mps" if torch.mps.is_available() and args.cuda else device)  # cuda and mps are taken to mean user wants to use the gpu
    
    if args.model_path == "last":
        args.model_path = "checkpoints/last_epoch.pt"
    elif args.model_path == "best":
        args.model_path = "checkpoints/val_loss_best.pt"

    if args.mode == "train":
        print("Training model...")
        train_ability_level_model(args, 
                                  train_loader=train, 
                                  validation_loader=val, 
                                  epochs=args.epochs, 
                                  num_question_range=args.question_range, 
                                  model_name=args.model_path,
                                  lr=args.lr,
                                  device=device,
                                  eval_intervals=args.eval_intervals)
    elif args.mode == "test":
        print("Testing model...")

        if not os.path.exists('graphs'):
                os.makedirs('graphs')
        if not os.path.exists("graphs/img"):
            os.makedirs("graphs/img")
        if not os.path.exists("graphs/html"):
            os.makedirs("graphs/html")

        if args.model_path is None:
            args.model_path = "checkpoints/val_loss_best.pt"
        model = RegressionModel().to(torch.device(device))
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        data.shuffle = True

        fig, ax = plt.subplots()
        line1, = ax.plot([], [], label='ML Estimator', color='blue')
        line2, = ax.plot([], [], label='EAP Estimator', color='orange')
        fill_ML = fill_EAP = None

        fig_2, ax_2 = plt.subplots()
        line3, = ax_2.plot([], [], label='p-values', color='red')
        line4, = ax_2.plot([], [], label='Significance Level (α=0.05)', color='black', linestyle='--')

        pbar_test = tqdm.tqdm(range(args.test_question_range[0], args.test_question_range[1]+1))
        ML_bounds = [[], [], []]
        EAP_bounds = [[], [], []]
        num_questions = []
        p_values = []
        for i in pbar_test:
            num_questions.append(i)
            TEST_ESTIMATES = [[], [], []]
            TEST_LOSSES = [[], []]
            avg_test_loss = 0.0
            avg_true_func_loss = 0.0
            pbar_epoch = tqdm.tqdm(range(args.test_epochs))
            for j in pbar_epoch:
                val_loss, true_loss = test_func(args, model, data, [i, i], torch.device(device), args.test_loss)
                avg_test_loss += val_loss
                avg_true_func_loss += true_loss
            avg_test_loss /= j+1
            avg_true_func_loss /= j+1

            # Calculate the tolerance interval for the test losses
            ML_bounds[0].append(avg_test_loss)
            EAP_bounds[0].append(avg_true_func_loss)
            sorted_ML_rmse = np.sort(TEST_LOSSES[0])
            sorted_EAP_rmse = np.sort(TEST_LOSSES[1])

            confidence = 0.95
            proportion = 0.90
            v = len(sorted_ML_rmse) - binom.ppf(confidence, len(sorted_ML_rmse), proportion)
            upper_index = np.ceil(len(sorted_ML_rmse) + 1 - v/2).astype(int)
            lower_index = np.floor(v/2).astype(int)

            ML_bounds[1].append(sorted_ML_rmse[lower_index])
            ML_bounds[2].append(sorted_ML_rmse[upper_index])
            EAP_bounds[1].append(sorted_EAP_rmse[lower_index])
            EAP_bounds[2].append(sorted_EAP_rmse[upper_index])

            plt.figure(1)

            line1.set_data(num_questions, ML_bounds[0])
            line2.set_data(num_questions, EAP_bounds[0])

            if fill_ML: fill_ML.remove()
            if fill_EAP: fill_EAP.remove()
            fill_ML = ax.fill_between(num_questions, ML_bounds[1], ML_bounds[2], color='blue', alpha=0.2)
            fill_EAP = ax.fill_between(num_questions, EAP_bounds[1], EAP_bounds[2], color='orange', alpha=0.2)

            ax.relim()
            ax.autoscale_view()
            ax.set_xlabel("Number of Random Questions")
            ax.set_ylabel(args.test_loss)
            ax.set_title(f"Performance of ML and EAP Estimators \n(Tolerance Interval: p={proportion}, 1-α={confidence})")
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1])
            ax.grid(True)
            ax.set_ylim(0, 2)
            
            fig.savefig("graphs/img/performance_view.png", dpi=300, bbox_inches='tight')

            # compute wilcoxon test to mark the difference between the two models
            stat, p = wilcoxon(TEST_LOSSES[0], TEST_LOSSES[1])
            p_values.append(p)

            reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

            plt.figure(2)
            line3.set_data(num_questions, pvals_corrected)
            line4.set_data(num_questions, [0.05] * len(num_questions))

            ax_2.relim()
            ax_2.autoscale_view()
            ax_2.set_xlabel("Number of Random Questions")
            ax_2.set_ylabel("p-value")
            ax_2.set_title("Wilcoxon Signed-Rank Test p-values")
            ax_2.legend()
            ax_2.grid(True)
            fig_2.savefig("graphs/img/wilcoxon_test.png", dpi=300, bbox_inches='tight')
            

            """
            mean_test_losses = np.mean(TEST_LOSSES, axis=1)
            TEST_ESTIMATES = np.array(TEST_ESTIMATES).reshape(3, -1)
            #fig_estimates = ff.create_distplot(TEST_ESTIMATES, ["ML Model Ability Estimate", "EAP Ability Estimate", "Target Ability Level"], bin_size=0.1)
            fig_estimates = go.Figure()
            fig_estimates.add_trace(go.Histogram(x=TEST_ESTIMATES[2], name="Target Ability Level", histnorm='probability density', opacity=0.85, marker=dict(color='#00BA38'))) #green
            fig_estimates.add_trace(go.Histogram(x=TEST_ESTIMATES[0], name="ML Model Ability Estimate", histnorm='probability density', opacity=0.75, marker=dict(color='#56B4E9'))) #blue
            fig_estimates.add_trace(go.Histogram(x=TEST_ESTIMATES[1], name="EAP Ability Estimate", histnorm='probability density', opacity=0.6, marker=dict(color='#FF7F00'))) #orange
            fig_estimates.update_layout(title=f"Estimates for {i} Questions",
                                        xaxis_title="Ability Level",
                                        yaxis_title="Probability Density",
                                        barmode='overlay')
            fig_estimates.write_html(f"graphs/html/estimates_{i}.html")
            #print("Attempting to write image for", i)
            #fig_estimates.write_image("graphs/img/estimates_" + str(i) + ".pdf", engine="kaleido")
            #pio.write_image(fig_estimates, "graphs/img/estimates_" + str(i) + ".pdf")
            #print("Image written for", i)

            #fig_losses = ff.create_distplot(TEST_LOSSES, ["ML Model -> " + str(round(mean_test_losses[0], 3)), "EAP -> " + str(round(mean_test_losses[1], 3))], bin_size=0.1,)
            fig_losses = go.Figure()
            mean_loss_0 = round(mean_test_losses[0], 3)
            fig_losses.add_trace(go.Violin(x=TEST_LOSSES[0], name=f"ML Model Loss -> {args.test_loss}: {mean_loss_0:.3f}  ", jitter=0.2, marker=dict(color='#56B4E9'))) #blue
            mean_loss_1 = round(mean_test_losses[1], 3)
            fig_losses.add_trace(go.Violin(x=TEST_LOSSES[1], name=f"EAP Loss -> {args.test_loss}: {mean_loss_1:.3f}  ", jitter=0.2, marker=dict(color='#FF7F00'))) #orange
            fig_losses.update_traces(meanline_visible=True)
            fig_losses.update_layout(title=f"{args.test_loss} Losses for {i} Questions",
                                     legend=dict(traceorder='reversed'),)
            fig_losses.write_html(f"graphs/html/losses_{i}.html")
            #fig_losses.write_image("graphs/img/losses_" + str(i) + ".pdf", engine="kaleido")
            #"""

            # seeds:
            # 111111
            # 222222
            # 333333
