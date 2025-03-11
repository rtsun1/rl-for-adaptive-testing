import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from tqdm import tqdm

def evaluate(
    model,
    model_path: str,
    make_env: Callable,
    make_env_args: dict,
    eval_episodes: int,
    num_envs: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(**make_env_args) for _ in range(num_envs)], 
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
        # shared_memory=True,
        copy=False,
    )
    if model == None:
        model = Model(envs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    thetas = []
    true_thetas = []
    with tqdm(total=eval_episodes, desc="Evaluating", unit="episode", leave=False) as pbar:
        while len(episodic_returns) < eval_episodes:
            if random.random() < epsilon:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                q_values = model(torch.Tensor(obs).to(device))
                actions = (
                    torch.argmax(q_values, dim=1).cpu().numpy()  # first dim is num envs. second dim is num actions
                )
            next_obs, reward, terminations, truncations, infos = envs.step(actions)

            done_mask = np.logical_or(terminations, truncations)

            if done_mask.any():
                n_done = done_mask.sum()
                returns = infos["episode"]["r"][done_mask]
                for i in range(n_done):
                    thetas.append(infos["theta"][done_mask][i])
                    true_thetas.append(infos["true_theta"][done_mask][i])
                    tqdm.write(
                        f"eval_episode={len(episodic_returns)}, episodic_return={returns[i]}"
                    )
                    episodic_returns.append(returns[i])
                    if len(episodic_returns) == eval_episodes:
                        break  # do not add extra returns if 2 epsiodes completed at the same time and needed only 1 more
                    pbar.update(1)
                obs, _ = envs.reset(options={"mask": done_mask})
            else:
                obs = next_obs

    histories = infos["action_history"]
    continuous_history = []
    for history in histories:
        continuous_history += history
    print(continuous_history)
    envs.close()
    # Done eval, go back to train
    model.train()

    return episodic_returns, thetas, true_thetas, continuous_history


if __name__ == "__main__":
    from dqn import QNetwork, make_env
    from dataloader import dataset_rl

    model_path = "runs/CAT-v0__dqn__42__1741654768/dqn.cleanrl_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, test_data = dataset_rl()
    make_env_args = {
        "env_id": "CAT-v0",
        "seed": 42,
        "dataset": val_data,
        "time_limit_s": 60 * 15,  # 15 minutes
        "include_answered": False,  # modify based on model
        "detailed_answered": False,  # modify based on model,
        "fisher_rewards": False,  # doesn't matter for evaluation
    }
    episodic_returns, thetas, true_thetas, histories = evaluate(
        model_path,
        make_env,
        make_env_args,
        eval_episodes=1000,
        num_envs=2,
        Model=QNetwork,
        device=device,
        epsilon=1,
    )
    print(episodic_returns)

    thetas = np.array(thetas)
    true_thetas = np.array(true_thetas)
    print("Estimated Thetas:")
    print(thetas.round(2))
    print("True Thetas:")
    print(true_thetas.round(2))
    rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
    
    print("RSME: ", rmse)