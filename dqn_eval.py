import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from tqdm import tqdm
from torch import nn

class Evaluator:
    def evaluate(
            self,
            agent,
            envs: gym.vector.VectorEnv,
            eval_episodes: int,
            epsilon: float = 0.05,
            verbose: bool = True,
        ):
        num_envs = envs.num_envs
        agent.eval()

        obs, infos = envs.reset()

        episodic_returns = []
        thetas = []
        true_thetas = []
        histories = []  # track the actions for each completed episode
        history = [[] for _ in range(num_envs)]  # track the actions for each episode in each env

        with tqdm(total=eval_episodes, desc="Evaluating", unit="episode", leave=True, disable=not verbose) as pbar:
            while len(episodic_returns) < eval_episodes:
                actions = agent.get_action(obs, infos, epsilon=epsilon)
                next_obs, reward, terminations, truncations, infos = envs.step(actions)

                # record action
                for env_idx in range(num_envs):
                    if not infos["already_asked"][env_idx]:
                        history[env_idx].append(actions[env_idx])

                done_mask = np.logical_or(terminations, truncations)

                if done_mask.any():
                    n_done = done_mask.sum()
                    for done_env in np.nonzero(done_mask)[0]:
                        thetas.append(infos["theta"][done_env])
                        true_thetas.append(infos["true_theta"][done_env])
                        # tqdm.write(
                        #     f"eval_episode={len(episodic_returns)}, episodic_return={infos["episode"]["r"][done_env]}"
                        # )
                        episodic_returns.append(infos["episode"]["r"][done_env])

                        histories.append(history[done_env])
                        history[done_env] = []

                        if len(episodic_returns) == eval_episodes:
                            break  # do not add extra returns if 2 epsiodes completed at the same time and needed only 1 more
                        pbar.update(1)

                    obs, _ = envs.reset(options={"mask": done_mask})
                else:
                    obs = next_obs

        # Done eval, go back to train
        agent.train()

        return episodic_returns, thetas, true_thetas, histories, False  # False means this run is never pruned


if __name__ == "__main__":
    from dqn import QNetwork, make_env, Agent
    from dataloader import dataset_rl

    train_data, val_data, test_data = dataset_rl()
    
    make_env_args = {
        "env_id": "CAT-v0",
        "seed": 42,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dataset": val_data,
        "time_limit_s": 60 * 15,  # 15 minutes
        "mask_answered": False,  # modify based on model
        "include_answered": False,  # modify based on model
        "detailed_answered": False,  # modify based on model,
        "fisher_rewards": False,  # doesn't matter for evaluation
        "ml_theta": False,
    }
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(**make_env_args)
            for i in range(4)
        ],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,  # see https://farama.org/Vector-Autoreset-Mode
        copy=False,
    )
    model_path = "runs/CAT-v0__dqn_mask_125000_train_steps_100_students__42__1741739511/dqn_mask_125000_train_steps_100_students.cleanrl_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(envs, make_env_args["mask_answered"])
    model.load_state_dict(torch.load(model_path))
    agent = Agent(model=model, device=device, envs=envs, mask_answered=True)

    e = Evaluator()
    episodic_returns, thetas, true_thetas, histories = e.evaluate(
        agent=agent,
        envs=envs,
        eval_episodes=1000,
        epsilon=0.05,
    )
    envs.close()
    print(episodic_returns)

    thetas = np.array(thetas)
    true_thetas = np.array(true_thetas)
    print("Estimated Thetas:")
    print(thetas.round(2))
    print("True Thetas:")
    print(true_thetas.round(2))
    rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
    
    print("RSME: ", rmse)