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

N_TRIALS = 100  # max num of trials
N_STARTUP_TRIALS = 5  # num of random sampling trials before using TPE sampler
N_EVALUATIONS = 3  # num of evaluations per trial
N_TIMESTEPS = 50_000  # num of steps per trial
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)  # num of steps before evaluating the agent
N_EVAL_EPISODES = 100  # num of episodes used for evaluating the agent

ENV_ID = "CAT-v0"
EVAL_EPSILON = 0.05
CUDA = False

def sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters from defined ranges."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    # max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    # gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    # n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    # learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    # ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    # ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    # net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    # trial.set_user_attr("gae_lambda_", gae_lambda)
    # trial.set_user_attr("n_steps", n_steps)

    # net_arch = [
    #     {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    # ]

    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        # "n_steps": n_steps,
        "gamma": gamma,
        # "gae_lambda": gae_lambda,
        # "learning_rate": learning_rate,
        # "ent_coef": ent_coef,
        # "max_grad_norm": max_grad_norm,
        # "policy_kwargs": {
        #     "net_arch": net_arch,
        #     "activation_fn": activation_fn,
        #     "ortho_init": ortho_init,
        # },
    }


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
            epsilon: float = 0.05
        ):
        episodic_returns, thetas, true_thetas, histories, pruned = super().evaluate(agent, envs, eval_episodes, epsilon)
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
    kwargs = sample_params(trial)
    # Create the args used to build and train the model.
    args = Args(**kwargs)
    args.cuda = CUDA
    args.env_id = ENV_ID
    args.total_timesteps = N_TIMESTEPS
    args.eval_epsilon = EVAL_EPSILON
    args.eval_episodes = N_EVAL_EPISODES
    args.eval_frequency = EVAL_FREQ
    args.seed = np.random.randint(2 ** 32 - 1)

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvaluator(trial)

    nan_encountered = False
    try:
        learn(args, eval_callback)
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

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
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