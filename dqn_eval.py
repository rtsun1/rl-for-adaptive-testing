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
                    if (len(episodic_returns) < eval_episodes):
                        obs, _ = envs.reset(options={"mask": done_mask})
                else:
                    obs = next_obs

        # Done eval, go back to train
        agent.train()

        return episodic_returns, thetas, true_thetas, histories, False  # False means this run is never pruned


if __name__ == "__main__":
    from dqn import QNetwork, make_env, Agent, MaskAgent
    from dataloader import dataset_rl
    for minutes in range(1, 33, 3):
        train_data, val_data, test_data = dataset_rl()

        time_limit = 60 * minutes  # 15 minutes

        make_env_args = {
            "env_id": "CAT-v0",
            "device": torch.device("cuda"),
            "time_limit_s": time_limit, 
            "mask_answered": True,  # modify based on model
            "include_answered": False,  # modify based on model
            # "include_uncertainty": False,
            "detailed_answered": False,  # modify based on model,
            "fisher_rewards": True,  # doesn't matter for evaluation
            "sparse_MSE_rewards":False,
            "ml_theta": True,
        }
        seed = np.random.randint(0, 2**32 - 1)
        envs = gym.vector.SyncVectorEnv(  # Async breaks ml theta state (using cuda in forked subprocess) in linux
            [
                make_env(seed=seed, dataset=test_data, **make_env_args)
                for i in range(1)
            ],
            autoreset_mode=gym.vector.AutoresetMode.DISABLED,  # see https://farama.org/Vector-Autoreset-Mode
            copy=False,
        )
        model_path = "runs/CAT-v0__dqn_fresh_new_ml__1926831558__1744018196/dqn_fresh_new_ml.cleanrl_model"
        device = torch.device("cuda")
        model = QNetwork(envs, make_env_args["mask_answered"])
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        agent = MaskAgent(model=model, target=model, device=device, envs=envs)

        e = Evaluator()
        episodic_returns, thetas, true_thetas, histories, _ = e.evaluate(
            agent=agent,
            envs=envs,
            eval_episodes=4272,
            epsilon=0,
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

        residuals = thetas - true_thetas
        np.savetxt(f"out/residuals_{time_limit}.csv", residuals)
        
        print("RMSE: ", rmse)