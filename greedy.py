import cat_env
import gymnasium as gym
from dataloader import dataset_rl
from IRT import batch_fisher_information
import time
from dqn import get_mask
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    save_results: bool = True
    """whether to save results into the `runs/{run_name}` folder"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Environment specific arguments
    time_limit_s: int = 60 * 15
    """the time limit for each episode/student in seconds"""

    # Algorithm specific arguments
    env_id: str = "CAT-v0"
    """the id of the environment"""
    time_aware: bool = False
    """whether to divide the Fisher Information by the average time to answer the question"""

    # Evaluation specific arguments
    eval_episodes: int = 100
    """the number of episodes/students to evaluate"""

def greedy_fi(args: Args):
    """
    Greedy Fisher Information Algorithm
    
    Parameters:
        time_aware (bool): Whether to divide the Fisher Information by the average time to answer the question
        n (int): Number of students to simulate
    
    Returns:
        thetas (np.array): Estimated thetas for each student
        true_thetas (np.array): True thetas for each student
    """
    n = args.eval_episodes
    time_aware = args.time_aware

    train_data, val_data, test_data = dataset_rl()
    env = gym.make("CAT-v0", dataset=val_data, time_limit_s=args.time_limit_s)  # 15 minutes
    # Greedy FI Algorithm
    obs, info = env.reset()

    n_items = info["dataset"]["item_id"].unique().shape[0]
    diffs = info["dataset"]["difficulty"].unique()  # unique preserves order of appearance
    discs = info["dataset"]["discrimination"].unique()
    thetas = np.zeros(n)  # estimated thetas
    true_thetas = np.zeros(n)

    for x in range(n):  # n students
        observation, info = env.reset(seed=42 + x)  # info gathered each reset
        # print(observation)
        theta = observation["theta"][0]
        fisher_inf = np.zeros(n_items)
        episode_over = False
        G = 0
        answered = set()

        ### TRAINING the short Theta over 1 episode
        while not episode_over: 

            # Calculate FI for all items
            # toc = time.time()
            fisher_inf = batch_fisher_information(diffs, discs, theta)
            mask = get_mask(n_items, [answered])  # get_mask assumes answered is from a vector env so wrap with []
            fisher_inf[np.logical_not(mask)] = -np.inf
            # tic = time.time()
            # print(tic - toc)
            if time_aware:
                fisher_inf /= info["average_times"]  # divide by avg time to answer the question
            # Select the action with the highest FI
            action = np.argmax(fisher_inf)
            answered.add(action)
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Update theta
            theta = observation["theta"][0]
            G += reward
            # Log
            print(action, observation, reward)
            
            episode_over = terminated or truncated
        
        # calculate true theta estimate for the current student
        # add to items
        thetas[x] = info["theta"]
        true_thetas[x] = info["true_theta"]

    env.close()
    return thetas, true_thetas


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)

    thetas, true_thetas = greedy_fi(
        args
    )

    # Theta RMSE:
    print("Estimated Thetas:")
    print(thetas.round(2))
    print("True Thetas:")
    print(true_thetas.round(2))
    rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
    
    print("RMSE: ", rmse)

    # Save the run
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    os.makedirs(f"runs/{run_name}", exist_ok=True)
    np.savez(
        f"runs/{run_name}/results.npz",
        thetas=thetas,
        true_thetas=true_thetas,
        rmse=rmse,
        args=args,
    )
    print(f"Results saved in runs/{run_name}.npz")
    print("Done!")
