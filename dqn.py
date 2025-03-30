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

from cat_wrappers import AnsweredVectorObservation, DetailedAnsweredVectorObservation, RecordActions, FisherInfoRewards, MaskObservation, MLThetaObservationAndInfo, UncertaintyVectorObservation

from typing import Callable
from tqdm import tqdm
from dqn_eval import Evaluator

import stable_baselines3 as sb3
from dataloader import dataset_rl

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
    wandb_entity: str = "b401p7"
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    log_interval: int = 5000
    """the interval between logging phase runtimes"""
    verbose: bool = True
    """whether to print logs"""

    # Environment specific arguments
    time_limit_s: int = 60 * 15
    """the time limit for each episode/student in seconds"""

    # Algorithm specific arguments
    env_id: str = "CAT-v0"
    """the id of the environment"""
    include_answered: bool = False
    """whether to include answered questions as a bit vector in the observation"""
    include_uncertainty: bool = False
    """whether to include uncertainty in ability estimate in the observation"""
    detailed_answered: bool = False
    """whether to include answered questions as a detailed vector (disc, diff, response, response time) in the observation"""
    mask_answered: bool = False
    """whether to mask actions that are already answered to -np.inf"""
    reward_function: str = "sparse"
    """reward function to use. one of ["sparse", "fisher"]"""
    theta_state: str = "irt"
    """the state representation to use. one of ["irt", "ml"]"""
    # dataset_f: str = "data/test_data.csv"
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    lr_scheduler: str = "Constant"
    """the learning rate scheduler to use, one of ["ReduceLROnPlateau", "IncreaseLROnPlateau", "CosineAnnealing", "Constant"]"""
    optimizer: str = "adam"
    """the optimizer to use, one of ["adam", "adamw"]"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 1
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 10000
    """the timesteps it takes to update the target network"""
    batch_size: int = 1024
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    e_scheduler: str = "linear"
    """the scheduler for epsilon, one of ["linear", "exponential"]"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    loss_function: str = "mse"
    """the loss function to use, one of ["mse", "l1", "huber"]"""

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
        for env in range(mask.shape[0]):  # iterating over envs so shouldn't be too slow
            mask[env, answered[env]] = 0
    else:  # case: making a mask of shape (n_items)
        mask[answered] = 0
    return mask


def make_env(
        env_id, seed, dataset, device,
        mask_answered, include_answered, include_uncertainty, detailed_answered, fisher_rewards, ml_theta,
        time_limit_s=60 * 15
    ):
    def thunk():
        env = gym.make(env_id, dataset=dataset, time_limit_s=time_limit_s)
        if include_answered and detailed_answered:
            raise ValueError("You are including both binary answered and detailed answered. This is likely a mistake.")
        if include_answered:
            env = AnsweredVectorObservation(env)
        if include_uncertainty:
            env = UncertaintyVectorObservation(env, dataset=dataset) # NOTE : don't hard code theta range
        elif detailed_answered:
            env = DetailedAnsweredVectorObservation(env, dataset=dataset)
        if fisher_rewards:
            env = FisherInfoRewards(env, dataset=dataset)
        if ml_theta:
            env = MLThetaObservationAndInfo(env, dataset=dataset, device=device)
        # env = RecordActions(env)  # High overhead in AsyncVectorEnv, so we will record actions in the main loop instead. Also messes up new evaluate implementation.
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        if mask_answered:
            env = MaskObservation(env)
        env.action_space.seed(seed)  # for random sampling actions
        return env

    return thunk

class Agent:
    def __init__(self, model: nn.Module, target: nn.Module, device, envs: gym.vector.VectorEnv):
        self.model = model
        self.target = target
        self.device = device
        self.envs = envs

    def get_action(self, obs, infos, epsilon=0.05):
        """Given an observation, choose an action."""
        if random.random() < epsilon:
            mask = np.ones((self.envs.num_envs, self.envs.single_action_space.n), dtype=np.int8)
            action = self.envs.action_space.sample(mask=tuple(mask[i] for i in range(mask.shape[0])))
        else:
            q_values = self.model(torch.Tensor(obs).to(self.device))
            action = torch.argmax(q_values, dim=1).cpu().numpy()  # first dim is num envs. second dim is num actions

        return action
    
    def get_target_values(self, obs):
        return self.target(obs)
    
    def get_values(self, obs):
        return self.model(obs)

    # def update(self, obs, actions, rewards, next_obs, dones):
    #     """Update the model with the observed results."""
        
    def train(self):
        """Set the model to train mode."""
        self.model.train()
    
    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()

class MaskAgent(Agent):
    def get_action(self, obs, infos, epsilon=0.05):
        if random.random() < epsilon:
            mask = np.logical_not(obs["answered"]).astype(np.int8)  # masking, gymnasium expects int8
            action = self.envs.action_space.sample(mask=tuple(mask[i] for i in range(mask.shape[0])))
        else:
            q_values = self.model(torch.Tensor(obs["main"]).to(self.device))
            mask = obs["answered"].astype(bool)  # masking, torch tensor expects bool
            q_values[mask] = -np.inf
            action = torch.argmax(q_values, dim=1).cpu().numpy()  # first dim is num envs. second dim is num actions 

        return action
    
    def get_target_values(self, obs):
        # important: max a' Q(S',a') needs to be masked aswell!
        next_q_values = self.target(obs["main"])
        mask = obs["answered"].bool()  # torch tensor expects bool
        next_q_values[mask] = -np.inf

        return next_q_values
    
    def get_values(self, obs):
        return self.model(obs["main"])

# TODO : Change QNetwork according to paper spec
# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, mask_answered=False):
        super().__init__()

        # print("DEBUG single_observation_space length: ", len(env.single_observation_space))
        # print("DEBUG action_space length: ", env.action_space.n)

        if mask_answered:
            input_size = np.array(env.single_observation_space["main"].shape).prod()
        else:
            input_size = np.array(env.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 240),
            nn.ReLU(),
            nn.Linear(240, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(
            x
        )  # TODO: implement masking here to allow for differentiable masking

# SCHEDULER :  for epsilon (exploration)
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
def exponential_schedule(start_e: float, end_e: float, duration: int, t: int):
    decay_rate = (end_e / start_e) ** (1 / duration)
    return max(start_e * (decay_rate ** t), end_e)

def learn(args: Args, evaluator: Evaluator = None, dict_args = {}):
    """
    The main function to learn the model.

    :param args: the arguments to run the experiment
    :param evaluator: the evaluation class to evaluate the model. 
    :param dict_args: arguments to pass that can be python objects (not just CLI args)
    Outputs whether to this hyperparameter setting is pruned or not.
    """
    if evaluator is None:
        evaluator = Evaluator()
    pruned = False
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            reinit=True
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
    device = torch.device("mps" if torch.mps.is_available() and args.cuda else device)  # cuda and mps are taken to mean user wants to use the gpu
    print("Using Device : ", device)

    # env setup
    main_env_config = {
            "env_id": args.env_id,
            "time_limit_s": 60.0 * 15,  # 15 minutes
            "device": device,
            "mask_answered": args.mask_answered,
            "include_answered": args.include_answered,
            "include_uncertainty": args.include_uncertainty,
            "detailed_answered": args.detailed_answered,
            "fisher_rewards": args.reward_function == "fisher",
            "ml_theta": args.theta_state == "ml",
        }  # just need to specify: seed, dataset
    
    # 4 envs
    # Sync: env step time ~= 0.0007 (NOTE: with record actions and using action_history for dqn_eval, was 0.0001 somehow...)
    # Async: env step time ~= 0.0004
    envs = gym.vector.SyncVectorEnv(  # Async breaks ml theta state (using cuda in forked subprocess) in linux
        [
            make_env(seed=args.seed + i, dataset=train_data, **main_env_config)
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,  # see https://farama.org/Vector-Autoreset-Mode
        copy=False,
    )
    # 4 envs, M2 chip:
    # Sync: ~33 ep/s
    # Async: ~20 ep/s
    eval_envs = gym.vector.SyncVectorEnv(
        [
            make_env(seed=args.seed + i, dataset=val_data, **main_env_config)
            for i in range(args.num_envs)
        ],
        copy=False,
    )
    # env = gym.make(args.env_id, dataset_f=args.dataset_f)
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    q_network = QNetwork(envs, args.mask_answered).to(device)
    if args.optimizer == "adam":
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(q_network.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Only adam and adamw are supported as optimizers")
    
    match args.lr_scheduler:
        case "ReduceLROnPlateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        case "IncreaseLROnPlateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
        case "CosineAnnealing":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.total_timesteps - args.learning_starts) / args.train_frequency)
        case "Constant":
            lr_scheduler = None
        case _:
            raise ValueError("Only ReduceLROnPlateau, IncreaseLROnPlateau, CosineAnnealing, and Constant are supported as lr_schedulers")
    target_network = QNetwork(envs, args.mask_answered).to(device)
    target_network.load_state_dict(q_network.state_dict())

    if args.mask_answered:
        agent = MaskAgent(model=q_network, target=target_network, device=device, envs=envs)
    else:
        agent = Agent(model=q_network, target=target_network, device=device, envs=envs)

    # print("DEBUG args.buffer_size : ", args.buffer_size)
    # print("DEBUG env.single_observation_space : ", env.single_observation_space)
    # print("DEBUG env.single_action_space : ", env.single_action_space)

    if args.mask_answered:
        replay_buffer = DictReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            envs.num_envs,
            handle_timeout_termination=False,
        )
    else:
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

    match args.loss_function:
        case "mse":
            loss_fn = F.mse_loss
        case "l1":
            loss_fn = F.l1_loss
        case "huber":
            loss_fn = F.smooth_l1_loss
        case _:
            raise ValueError("Only mse, l1, and huber are supported as loss functions")

    if args.e_scheduler == "linear":
        e_scheduler = linear_schedule
    elif args.e_scheduler == "exponential":
        e_scheduler = exponential_schedule
    else:
        raise ValueError("Only linear and exponential are supported as e_schedulers")
    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset(seed=args.seed)  # TODO : Batched observations
    # flattened_obs = flatten(envs.single_observation_space, obs)  # MODIFICATION : dict
    step_counter = {"eval_episodes": 0}
    for global_step in tqdm(range(args.total_timesteps), disable=not args.verbose):
        # ALGO LOGIC: put action logic here
        epsilon = e_scheduler(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )


        start_forward_time = time.time()
        actions = agent.get_action(obs, infos, epsilon=epsilon)
        forward_times.append(time.time() - start_forward_time)
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

        # Get done envs
        done_mask = np.logical_or(terminations, truncations)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if done_mask.any():
            returns = infos["episode"]["r"][done_mask]
            episode_lengths = infos["episode"]["l"][done_mask]
            for i in range(returns.shape[0]):
                # tqdm.write(f"global_step={global_step}, episodic_return={returns[i]}")
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
                start_train_time = time.time()

                data = replay_buffer.sample(args.batch_size)
                with torch.no_grad():
                    next_q_values = agent.get_target_values(data.next_observations)
                    target_max, _ = next_q_values.max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max
                old_val = agent.get_values(data.observations).gather(1, data.actions).squeeze()
                loss = loss_fn(td_target, old_val)

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

                # print("DEBUG : loss ", loss)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_scheduler:
                    if args.lr_scheduler == "ReduceLROnPlateau" or args.lr_scheduler == "IncreaseLROnPlateau":
                        lr_scheduler.step(metrics=loss)
                    elif args.lr_scheduler == "CosineAnnealing":
                        lr_scheduler.step()

                train_times.append(time.time() - start_train_time)
                
            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data
                        + (1.0 - args.tau) * target_network_param.data
                    )
        # Periodically log time
        if global_step % args.log_interval == 0:  # TODO: remove hardcoded interval
            if args.verbose and len(train_times) > 0:
                tqdm.write(f"env_step_time={np.mean(env_times)}, reset_time={np.mean(reset_times)}, forward_times={np.mean(forward_times)}, train_time={np.mean(train_times)}")
            env_times = []
            train_times = []
            forward_times = []
            reset_times = []

        # PERIODICALLY EVAL
        if global_step % args.eval_frequency == 0:
            episodic_returns, thetas, true_thetas, histories, pruned = evaluator.evaluate(
                agent=agent,
                envs=eval_envs,
                eval_episodes=args.eval_episodes,
                epsilon=args.eval_epsilon,
                verbose=args.verbose
            )
            # for episodic_return in episodic_returns:
            #     writer.add_scalar("eval/episodic_return", episodic_return, global_step)
            #     step_counter["eval_episodes"] += 1
            writer.add_scalar("eval/avg_episodic_return", np.mean(episodic_returns), global_step)
            thetas = np.array(thetas)
            true_thetas = np.array(true_thetas)
            rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
            writer.add_scalar("eval/RMSE", rmse, global_step)

            if pruned:
                break

    # Final evaluation
    if not pruned:
        episodic_returns, thetas, true_thetas, histories, pruned = evaluator.evaluate(
            agent=agent,
            envs=eval_envs,
            eval_episodes=args.eval_episodes,
            epsilon=args.eval_epsilon,
            verbose=args.verbose
        )
        # for idx, episodic_return in enumerate(episodic_returns):
        #     writer.add_scalar("eval/episodic_return", episodic_return, idx)
        writer.add_scalar("eval/avg_episodic_return", np.mean(episodic_returns), global_step)
        thetas = np.array(thetas)
        true_thetas = np.array(true_thetas)
        rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
        writer.add_scalar("eval/RMSE", rmse, global_step)
        if args.verbose:
            print("Estimated Thetas:")
            print(thetas.round(2))
            print("True Thetas:")
            print(true_thetas.round(2))
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

        if args.verbose:
            print(f"Results saved in runs/{run_name}/results.npz")
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(q_network.state_dict(), model_path)
            if args.verbose:
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
        eval_envs.close()
        writer.close()

        if args.track:
            run.finish()

if __name__ == "__main__":
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )
    args = tyro.cli(Args)
    learn(args)
