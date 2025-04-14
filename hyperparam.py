"""
Based off: 
https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on a Gymnasium environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.
"""

from typing import Any
from typing import Dict, Callable

import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn

from dqn import Args, learn
from dqn_eval import Evaluator
import os
import numpy as np

EXP_NAME = "dqn_fresh_new_ml"
N_TRIALS = 10_000  # max num of trials
N_STARTUP_TRIALS = 10  # num of random sampling trials before using TPE sampler
#N_EVALUATIONS = 6  # num of evaluations per trial
N_EVALUATIONS = 8  # num of evaluations per trial
N_TIMESTEPS = 1_000_000  # num of steps per trial
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)  # num of steps before evaluating the agent
N_EVAL_EPISODES = 1000  # num of episodes used for evaluating the agent

ENV_ID = "CAT-v0"
EVAL_EPSILON = 0
CUDA = True

def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters from defined ranges."""
    # === Not implemented hyperparameters: ===
    # net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])

    exp_name = EXP_NAME
    gamma = 1.0 #trial.suggest_float("gamma", 0.8, 1, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # === Problem formulation hyperparameters: ===
    include_answered: bool = False
    # include_answered = trial.suggest_categorical("include_answered", [True, False])
    detailed_answered: bool = False
    # detailed_answered = trial.suggest_categorical("detailed_answered", [True, False])
    mask_answered = True
    reward_function = trial.suggest_categorical("reward_function", ["fisher", "mse"])
    # reward_function = "fisher"
    theta_state = trial.suggest_categorical("theta_state", ["irt", "ml"])  # TODO: try true theta as well 
    # theta_state = "ml"

    # === DQN hyperparameters: ===
    loss_function = trial.suggest_categorical("loss_function", ["l1", "mse", "huber"])
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    lr_scheduler = trial.suggest_categorical("lr_scheduler", ["ReduceLROnPlateau", "IncreaseLROnPlateau", "CosineAnnealing", "Constant"])
    # lr_scheduler = trial.suggest_categorical("lr_scheduler", ["CosineAnnealing", "Constant"])
    e_scheduler = trial.suggest_categorical("epsilon_decay_scheduler", ["linear", "exponential"])
    num_envs = 1
    tau = 1.0
    # tau = trial.suggest_float("tau", 0.01, 1.0, log=True)
    target_network_frequency = trial.suggest_int("target_network_frequency", 1000, 50_000, log=True)
    # target_network_frequency = trial.suggest_int("target_network_frequency", 1000, 15_000, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 128, log=True)
    buffer_size: int = trial.suggest_int("buffer_size", max(5000, batch_size), 500_000, log=True)
    start_e = 1
    end_e = trial.suggest_float("end_e", 0.01, 0.1, log=True)
    exploration_fraction = 0.5
    learning_starts = trial.suggest_int("learning_starts", 1000, 100_000, log=True)
    # learning_starts = trial.suggest_int("learning_starts", max(1000, batch_size), 20_000, log=True)
    train_frequency = trial.suggest_int("train_frequency", 1, 64, log=True)
    # train_frequency = trial.suggest_int("train_frequency", 1, 100, log=True)

    # Display true values.
    # trial.set_user_attr("gamma_", gamma)
    # trial.set_user_attr("gae_lambda_", gae_lambda)
    # trial.set_user_attr("n_steps", n_steps)

    # net = [
    #     {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    # ]

    cli_args = {
        "exp_name": exp_name,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "loss_function": loss_function,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "e_scheduler": e_scheduler,
        # "net_arch": net,
        "include_answered": include_answered,
        "detailed_answered": detailed_answered,
        "theta_state": theta_state,
        "mask_answered": mask_answered,
        "reward_function": reward_function,
        "num_envs": num_envs,
        "buffer_size": buffer_size,
        "tau": tau,
        "target_network_frequency": target_network_frequency,
        "batch_size": batch_size,
        "start_e": start_e,
        "end_e": end_e,
        "exploration_fraction": exploration_fraction,
        "learning_starts": learning_starts,
        "train_frequency": train_frequency,
    }
    dict_args = {}
    return cli_args, dict_args


class TrialEvaluator(Evaluator):
    """
    Callback used for evaluating and reporting a trial.
    Also early stops the learn function if the trial is pruned.
    """
    def __init__(
        self,
        trial: optuna.Trial,
    ):
        # self.deterministic = deterministic
        # self.verbose = verbose
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

        self.n_calls = 0
        self.rmse = np.inf

    def evaluate(
            self, 
            agent,
            envs: gym.vector.VectorEnv,
            eval_episodes: int,
            epsilon: float = 0.05,
            verbose=True
        ):
        episodic_returns, thetas, true_thetas, histories, pruned = super().evaluate(agent, envs, eval_episodes, epsilon, verbose)
        thetas = np.array(thetas)
        true_thetas = np.array(true_thetas)
        self.rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
        
        self.eval_idx += 1
        self.trial.report(self.rmse, self.eval_idx)

        # Prune trial if need.
        if self.trial.should_prune():
            self.is_pruned = True

        return episodic_returns, thetas, true_thetas, histories, self.is_pruned

def objective(trial: optuna.Trial) -> float:
    # Sample hyperparameters. TODO:
    kwargs, dict_args = sample_params(trial)
    # Create the args used to build and train the model.
    args = Args(**kwargs)
    args.cuda = CUDA
    args.env_id = ENV_ID
    args.total_timesteps = N_TIMESTEPS
    args.eval_epsilon = EVAL_EPSILON
    args.eval_episodes = N_EVAL_EPISODES
    args.eval_frequency = EVAL_FREQ
    args.seed = np.random.randint(np.iinfo(np.int32).max)
    # Turn off logging in a trial
    args.verbose = False
    args.track = True

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvaluator(trial)

    nan_encountered = False
    try:
        learn(args, eval_callback, dict_args)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.rmse


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_min_trials=N_EVALUATIONS // 3)

    # optuna.delete_study(study_name="dqn_masking", storage="sqlite:///studies.db")
    study = optuna.create_study(
        study_name=EXP_NAME, 
        storage="sqlite:///studies.db",
        load_if_exists=True,
        sampler=sampler, pruner=pruner, direction="minimize"
    )
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None, show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
