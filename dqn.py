# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import cat_env
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces.utils import flatten

from cat_wrappers import AnsweredVectorObservation, DetailedAnsweredVectorObservation, RecordActions, FisherInfoRewards

from tqdm import tqdm
from dqn_eval import evaluate


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "CAT_dqn"
    """the wandb's project name"""
    wandb_entity: str = "tonyinbio"
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Environment specific arguments
    time_limit_s: int = 60 * 15
    """the time limit for each episode/student in seconds"""

    # Algorithm specific arguments
    env_id: str = "CAT-v0"
    """the id of the environment"""
    include_answered: bool = False
    """whether to include answered questions as a bit vector in the observation"""
    detailed_answered: bool = False
    """whether to include answered questions as a detailed vector (disc, diff, response, response time) in the observation"""
    mask_answered: bool = False
    """whether to mask actions that are already answered to -np.inf"""
    fisher_rewards: bool = False
    """whether to add fisher information as rewards"""
    # dataset_f: str = "data/test_data.csv"
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    # Evaluation specific arguments
    eval_episodes: int = 100
    """the number of episodes to evaluate"""
    eval_epsilon: float = 0.05
    """the epsilon for evaluation"""
    eval_frequency: int = 50000


def get_mask(
    shape, answered
):  # Masking is differentiable: https://arxiv.org/pdf/2006.14171
    """
    Given a set of answered questions, 
    return a mask with 1 meaning the question has not been asked.
    Parameters:
        shape (ShapeLike): The desired shape of the returned mask.
        answered (iterable): is an array of sets of answered questions
    Returns:
        numpy.ndarray: An array with 1 for unanswered and 0 for answered items.
    """

    mask = np.ones(shape, dtype=np.int8)  # int8 b/c gym expects this
    answered = [list(a) for a in answered]
    if mask.ndim > 1:  # case: making a mask of shape (n_env, n_items)
        for i in range(mask.shape[0]):
            mask[i, answered[i]] = 0
    else:  # case: making a mask of shape (n_items)
        mask[answered] = 0
    return mask


def make_env(env_id, seed, dataset, include_answered, detailed_answered, fisher_rewards, time_limit_s=60 * 15):
    def thunk():
        env = gym.make(env_id, dataset=dataset, time_limit_s=time_limit_s)
        if include_answered and detailed_answered:
            raise ValueError("You are including both binary answered and detailed answered. This is likely a mistake.")
        if include_answered:
            env = AnsweredVectorObservation(env)
        elif detailed_answered:
            env = DetailedAnsweredVectorObservation(env, dataset=dataset)
        if fisher_rewards:
            env = FisherInfoRewards(env, dataset=dataset)
        # env = RecordActions(env)  # High overhead in AsyncVectorEnv, so we will record actions in the main loop instead
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        env.action_space.seed(seed)  # for random sampling actions
        return env

    return thunk


# TODO : Change QNetwork according to paper spec
# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        # print("DEBUG single_observation_space length: ", len(env.single_observation_space))
        # print("DEBUG action_space length: ", env.action_space.n)
        #input_size = np.array(env.single_observation_space.shape).prod()
        input_size = np.array(env.single_observation_space.shape).prod()
        output_size = env.single_action_space.n

        self.network = nn.Sequential(
            nn.Linear(input_size, input_size * 5),
            nn.ReLU(),
            nn.Linear(input_size * 5, input_size * 5),
            nn.ReLU(),
            nn.Linear(input_size * 5, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 6),
            nn.ReLU(),
            nn.Linear(input_size // 6, output_size * 5),
            nn.ReLU(),
            nn.Linear(output_size * 5, output_size),
        )
        """
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(),
            nn.Linear(input_size * 2, input_size * 1),
            nn.ReLU(),
            nn.Linear(input_size * 1, input_size // 6),
            nn.ReLU(),
            nn.Linear(input_size // 6, input_size // 6),
            nn.ReLU(),
            nn.Linear(input_size // 6, output_size * 10),
            nn.ReLU(),
            nn.Linear(output_size * 10, output_size),
            #"""

    def forward(self, x):
        return self.network(
            x
        )  # TODO: implement masking here to allow for differentiable masking


# SCHEDULER :  for epsilon (exploration)
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3
    from dataloader import dataset_rl

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train_data, val_data, test_data = dataset_rl()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("mps" if torch.mps.is_available() and args.cuda else "cpu")  # cuda and mps are taken to mean user wants to use the gpu
    device = torch.device("cuda")
    print("Using Device : ", device)

    # env setup
    main_env_config = {
            "env_id": args.env_id,
            "time_limit_s": 60.0 * 15,  # 15 minutes
            "include_answered": args.include_answered,
            "detailed_answered": args.detailed_answered,
            "fisher_rewards": args.fisher_rewards,
        }  # just need to specify: seed, dataset
    
    # 4 envs
    # Sync: env step time ~= 0.0007 (NOTE: with record actions and using action_history for dqn_eval, was 0.0001 somehow...)
    # Async: env step time ~= 0.0004
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(seed=args.seed + i, dataset=train_data, **main_env_config)
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,  # see https://farama.org/Vector-Autoreset-Mode
        copy=False,
    )
    # env = gym.make(args.env_id, dataset_f=args.dataset_f)
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # print("DEBUG args.buffer_size : ", args.buffer_size)
    # print("DEBUG env.single_observation_space : ", env.single_observation_space)
    # print("DEBUG env.single_action_space : ", env.single_action_space)

    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        envs.num_envs,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    env_times = []  # track time to step the env
    forward_times = []  # track time to forward the model
    train_times = []
    reset_times = []
    last_n = []

    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset(seed=args.seed)  # TODO : Batched observations
    # flattened_obs = flatten(envs.single_observation_space, obs)  # MODIFICATION : dict
    step_counter = {"eval_episodes": 0}
    avg_loss = 0
    loss_count = 0
    steps = tqdm(range(args.total_timesteps))
    for global_step in steps:
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        # print("DEBUG obs : ", obs)
        # print("DEBUG flattened_obs : ", flattened_obs)

        if random.random() < epsilon:
            if args.mask_answered:
                mask = get_mask((envs.num_envs, envs.single_action_space.n), infos["answered"])
                actions = envs.action_space.sample(mask=tuple(mask[i] for i in range(mask.shape[0])))
            else:
                actions = envs.action_space.sample()
            # print("DEBUG : global step ", global_step, " random ", actions)
        else:
            start_forward_time = time.time()

            q_values = q_network(torch.Tensor(obs).to(device))
            if args.mask_answered:
                mask = get_mask(q_values.shape, infos["answered"])  # masking
                q_values[np.logical_not(mask)] = -np.inf
            # print("DEBUG q_values  : ", q_values)
            actions = (
                torch.argmax(
                    q_values, dim=1
                )  # TODO : Check if modifying dim to 0 crashes batche
                .cpu()
                .numpy()
            )

            #forward_times.append(time.time() - start_forward_time)
            # print("DEBUG : global step ", global_step, " q-network ", actions)

        # print(
        #     "DEBUG actions : ",
        #     actions.shape,
        #     actions,
        #     type(actions),
        #     type(actions.item()),
        # )

        # TRY NOT TO MODIFY: execute the game and log data.
        start_env_time = time.time()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        env_times.append(time.time() - start_env_time)
        # print(rewards)
        # Get done envs
        done_mask = np.logical_or(terminations, truncations)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if done_mask.any():
            returns = infos["episode"]["r"][done_mask]
            episode_lengths = infos["episode"]["l"][done_mask]
            for i in range(returns.shape[0]):
                #tqdm.write(f"global_step={global_step}, episodic_return={returns[i]}")
                writer.add_scalar("training/episodic_return", returns[i], global_step)
                writer.add_scalar("training/episodic_length", episode_lengths[i], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # NOTE: Fixed based on: https://farama.org/Vector-Autoreset-Mode
        # real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        # replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)
        replay_buffer.add(
            obs, next_obs, actions, rewards, terminations, infos
        )  # TODO: Ensure sending terminations and not truncated is correct

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        if done_mask.any():
            start_reset_time = time.time()

            obs, _ = envs.reset(options={"mask": done_mask})

            reset_times.append(time.time() - start_reset_time)
        else:
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                #avg_loss = np.inf
                #last_n = []
                #while avg_loss > 30:
                start_train_time = time.time()

                data = replay_buffer.sample(args.batch_size)
                with torch.no_grad():
                    if args.mask_answered:
                        next_q_values = target_network(data.next_observations)
                        mask = get_mask(next_q_values.shape, data.infos["answered"])
                        next_q_values[np.logical_not(mask)] = -np.inf
                        target_max, _ = next_q_values.max(dim=1)
                    else:
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(old_val, td_target)
                
                last_n.append(loss.item())
                loss_count = len(last_n)
                if loss_count > 500:
                    last_n.pop(0)
                avg_loss = sum(last_n) / loss_count
                steps.set_description(f"Avg. Loss: {round(avg_loss, 3)}")

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #if global_step/args.train_frequency % 500 == 0:
                #    lr_scheduler.step(loss)

                train_times.append(time.time() - start_train_time)
                
            # update target network
            if global_step % args.target_network_frequency == 0:
                avg_loss = 0
                loss_count = 0
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data
                        + (1.0 - args.tau) * target_network_param.data
                    )
        # Periodically log time
        #if (global_step+1) % 5000 == 0:  # TODO: remove hardcoded interval
        #    tqdm.write(f"env_step_time={np.mean(env_times)}, reset_time={np.mean(reset_times)}, forward_times={np.mean(forward_times)}, train_time={np.mean(train_times)}")
        #    env_times = []
        #    train_times = []
        #    forward_times = []
        #    reset_times = []

        # PERIODICALLY EVAL
        if global_step % args.eval_frequency == 0 and global_step > args.train_frequency:
            make_env_args = {
                "seed": args.seed,  # for random sampling actions
                "dataset": val_data,
                **main_env_config,
            }
            episodic_returns, thetas, true_thetas, histories = evaluate(
                model=q_network,
                model_path="",
                make_env=make_env,
                make_env_args=make_env_args,
                eval_episodes=args.eval_episodes,
                num_envs=args.num_envs,
                Model=QNetwork,
                device=device,
                epsilon=args.eval_epsilon,
            )
            for episodic_return in episodic_returns:
                writer.add_scalar("eval/episodic_return", episodic_return, global_step)
                step_counter["eval_episodes"] += 1
            thetas = np.array(thetas)
            true_thetas = np.array(true_thetas)
            rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
            writer.add_scalar("eval/RMSE", rmse, global_step)

    make_env_args = {
        "seed": args.seed,  # for random sampling actions
        "dataset": val_data,
        **main_env_config,
    }
    episodic_returns, thetas, true_thetas, histories = evaluate(
        model=q_network,
        model_path="",
        make_env=make_env,
        make_env_args=make_env_args,
        eval_episodes=args.eval_episodes,
        num_envs=args.num_envs,
        Model=QNetwork,
        device=device,
        epsilon=args.eval_epsilon,
    )
    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)
    thetas = np.array(thetas)
    true_thetas = np.array(true_thetas)
    print("Estimated Thetas:")
    print(thetas.round(2))
    print("True Thetas:")
    print(true_thetas.round(2))
    rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
    writer.add_scalar("eval/RMSE", rmse, global_step)
    print("RMSE: ", rmse)

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    packaged_histories = np.zeros(1, dtype=object)
    packaged_histories[0] = histories
    np.savez(
        f"runs/{run_name}/results.npz",
        thetas=thetas,
        true_thetas=true_thetas,
        rmse=rmse,
        args=args,
        histories=packaged_histories,
    )
    print(f"Results saved in runs/{run_name}/results.npz")
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        print("Done!")

    #     if args.upload_model:
    #         from cleanrl_utils.huggingface import push_to_hub

    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(
    #             args,
    #             episodic_returns,
    #             repo_id,
    #             "DQN",
    #             f"runs/{run_name}",
    #             f"videos/{run_name}-eval",
    #         )

    envs.close()
    writer.close()
