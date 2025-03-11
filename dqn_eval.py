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
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
):
    # make_env_args["capture_video"] = make_env_args.get("capture_video", True)
    envs = gym.vector.SyncVectorEnv([make_env(**make_env_args)], gym.vector.AutoresetMode.DISABLED)
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
            # print(envs)
            # print(infos)
            # print(infos["student_id"][0], terminations[0], actions[0])

            if done_mask.any():
                thetas.append(infos["theta"][done_mask].squeeze())
                true_thetas.append(infos["true_theta"][done_mask].squeeze())

                returns = infos["episode"]["r"][done_mask]
                for i in range(returns.shape[0]):
                    tqdm.write(
                        f"eval_episode={len(episodic_returns)}, episodic_return={returns[i]}"
                    )
                    episodic_returns += [returns[i]]
                    pbar.update(1)
                obs, _ = envs.reset(options={"mask": done_mask})
            else:
                obs = next_obs

    histories = infos["histories"]
    envs.close()
    # Done eval, go back to train
    model.train()

    return episodic_returns, thetas, true_thetas, histories


if __name__ == "__main__":
    # from huggingface_hub import hf_hub_download

    from dqn import QNetwork, make_env
    from dataloader import dataset_rl
    model_path = "runs/CAT-v0__dqn_danic_rep_500000_train_steps_1000_students__42__1741685881/dqn_danic_rep_500000_train_steps_1000_students.cleanrl_model"
    # model_path = hf_hub_download(
    #     repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="q_network.pth"
    # )
    # evaluate(
    #     model_path,
    #     make_env,
    #     "CartPole-v1",
    #     eval_episodes=10,
    #     run_name=f"eval",
    #     Model=QNetwork,
    #     device="cpu",
    #     capture_video=False,
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, test_data = dataset_rl()
    make_env_args = {
        "env_id": "CAT-v0",
        "seed": 42,
        "dataset": val_data,
        "time_limit_s": 60 * 15,  # 15 minutes
        "include_answered": False,  # modify based on model
        "detailed_answered": True,  # modify based on model,
        "fisher_rewards": False,  # doesn't matter for evaluation
    }
    episodic_returns, thetas, true_thetas, histories = evaluate(
        model_path,
        make_env,
        make_env_args,
        eval_episodes=1000,
        Model=QNetwork,
        device=device,
        epsilon=0.05,
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
    # for idx, episodic_return in enumerate(episodic_returns):
        # writer.add_scalar("eval/episodic_return", episodic_return, idx)