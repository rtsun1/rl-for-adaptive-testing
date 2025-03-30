import gymnasium as gym
import numpy as np
from typing import Optional
import pandas as pd

from IRT import calculate_eap_ability, batch_fisher_information

class CATEnv(gym.Env):
    def __init__(
        self,
        dataset: pd.DataFrame,
        time_limit_s: float = 60.0 * 15,  # 15 minutes
        init_theta: float = 0.0,
        theta_lb: float = -4.0,
        theta_ub: float = 4.0,
    ):
        """
        Define the observation and action spaces

        Observation space: {"theta": float, "time_left": float}

        Action space: [0, MAX_Q]

        dataset: pd.DataFrame with columns ["student_id", "item_id", "discrimination", "difficulty", "response", "response_time", "final_score"]
        time_limit_s: max time for a test in seconds
        init_theta: initial ability estimate
        theta_lb: lower bound for theta
        theta_ub: upper bound for theta
        """
        super().__init__()  # may not be necessary, but it's good practice

        self.dataset = self._sort_dataset(dataset)
        
        items = self.dataset["item_id"].unique()
        self.n_items = items.shape[0]
        self.students = self.dataset["student_id"].unique()

        self.average_times = self.dataset.groupby("item_id")["response_time"].mean().sort_index().to_numpy()

        self.time_limit_s = time_limit_s
        self.init_theta = init_theta
        self.theta_range = (theta_lb, theta_ub)

        # State variables
        self.time_taken_s = 0
        self.theta = self.init_theta
        self.true_theta = 0  # init in reset
        self.sid = None  # active student id
        self.answered = set()  # items already answered
        self.responses = []  # student responses
        self.discrimination = []  # item discrimination
        self.difficulty = []  # item difficulty

        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "theta": gym.spaces.Box(
                    low=theta_lb, high=theta_ub, shape=(1,), dtype=np.float32
                ),  # shape (1,) [NOT ()] is default for Box space with scalar values
                "time_left": gym.spaces.Box(
                    low=0, high=self.time_limit_s, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Define action space
        self.action_space = gym.spaces.Discrete(self.n_items)

        self._action_to_idx = {
            action: id for action, id in enumerate(items)
        }

    def _sort_dataset(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the dataset (raw)

        return: DataFrame sorted by student_id and item_id
            cols: ["student_id", "item_id", "discrimination", "difficulty", "response", "response_time", "final_score"]
        """
        raw.sort_values(by=["student_id", "item_id"], inplace=True)  # so we can access the items in O(1) time
        raw.reset_index(drop=True, inplace=True)

        n_items = raw["item_id"].unique().shape[0]

        # NOTE: Assumptions for O(1) lookup: 1. Items are sorted by student_id and item_id, 2. Each student has the same number of items
        if raw.shape[0] % n_items != 0:
            raise ValueError("Each student must have the same number of items")

        return raw

    def _get_obs(self):
        """
        Get the current observation
        """
        time_left = self.time_limit_s - self.time_taken_s
        normalized_time_left = time_left / self.time_limit_s
        return {
            "theta": np.array(self.theta, dtype=np.float32).reshape(
                1
            ),  # NOTE: reshaping to (1,) from () to make it consistent with the Box space
            "time_left": np.array(
                normalized_time_left, dtype=np.float32
            ).reshape(1),
        }

    def _get_info(self):
        """
        Get the current info
        """
        return {
            "student_id": self.sid, 
            "theta": self.theta,
            "already_asked": self.action_already_asked,
            "true_theta": self.true_theta,
            "average_times": self.average_times,
            "answered": self.answered,
            "sid_idx": self.sid_idx,
        }
    
    def _get_true_theta(self):
        """
        Get the true theta for the student
        """
        return self.dataset.iloc[self.sid_idx * self.n_items]["final_score"]  # just get the precomputed final score
        # start = self.sid_idx * self.n_items
        # stop = (self.sid_idx + 1) * self.n_items

        # all_responses = self.dataset.iloc[start:stop][
        #     "response"
        # ].values
        # all_discrimination = self.dataset[start:stop][
        #     "discrimination"
        # ].values
        # all_difficulty = self.dataset[start:stop][
        #     "difficulty"
        # ].values

        # true_theta, _ = calculate_eap_ability(
        #     all_responses, all_discrimination, all_difficulty, self.theta_range
        # )

        # return true_theta
        

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Initialize a new episode

        Start state is no response and time taken is 0.
        Use seed to select the student.
        Options currently not used.
        """
        super().reset(seed=seed)  # For seeding self.np_random

        # Select a student
        self.sid_idx = self.np_random.integers(self.students.shape[0])
        self.sid = self.students[self.sid_idx]

        # Reset the state
        self.time_taken_s = 0
        self.theta = self.init_theta
        self.true_theta = self._get_true_theta()  # NOTE: probably takes the most time
        self.action_already_asked = False  # whether the last action was already asked
        self.answered = set()  # items already answered
        self.responses = []  # student responses
        self.discrimination = []  # item discrimination
        self.difficulty = []  # item difficulty

        # Return the observation and info
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Take a step in the environment.
        Update the state (theta and time left) as soon as we get the response.

        Returns:
            observation: current observation
            reward: reward for the action
            terminated: whether the episode is terminated
            truncated: whether the episode is truncated
            info: additional information
        """
        # If the item is already answered, return the same observation and info
        # Also make the reward negative to discourage the agent from selecting the same item
        # print("ANSWERED", self.answered)
        if action in self.answered:
            self.action_already_asked = True
            return (
                self._get_obs(),
                -1,
                False,
                False,
                self._get_info(),
            )  # NOTE: returning a reward of -1. might need to tune based on the reward function
        # NOTE: Another potential problem above is an infinite loop if the agent keeps selecting the same item (if this reward is better than normal rewards)
        self.action_already_asked = False
        
        # Get the index range for student (because dataset is sorted by student_id and item_id)
        start = self.sid_idx * self.n_items

        # Otherwise, get the item data
        item = self._action_to_idx[action]
        item_data = self.dataset.iloc[start + action]
        if item_data.empty:
            raise ValueError(
                f"Action {action} aka Item ID {item} not found in the dataset. For student ID {self.sid}"
            )
        
        # Extract the item parameters
        discrimination = item_data["discrimination"]
        difficulty = item_data["difficulty"]
        response = item_data["response"]
        response_time = item_data["response_time"]
        # print(f"Action {action}: time: {response_time}")

        # If the student couldn't answer the question before time ran out, terminate the episode
        self.time_taken_s += response_time
        if self.time_taken_s > self.time_limit_s:
            self.time_taken_s = self.time_limit_s  # cap the time to be within the observation space
            # Multiply by 50 to scale the reward relative to -1 reward for answering the same question
            # reward = 50 * - (self.true_theta - self.theta) ** 2  # NOTE: reward is the negative of the squared error in the ability estimate
            reward = 0
            return self._get_obs(), reward, True, False, self._get_info()  # return the error in the ability estimate (from the entire dataset vs the student's responses)
        
        # Otherwise, update the state
        self.answered.add(action)
        self.responses.append(response)
        self.discrimination.append(discrimination)
        self.difficulty.append(difficulty)

        # Calculate the EAP ability
        self.theta, _ = calculate_eap_ability(
            self.responses, self.discrimination, self.difficulty, self.theta_range
        )

        # Calculate the reward
        reward = (
            0  # NOTE: Currently, reward 0 for all steps until the episode is terminated
        )

        return self._get_obs(), reward, False, False, self._get_info()

    def render(self):
        """
        Render the environment
        """
        pass

    def close(self):
        """
        Close the environment
        """
        pass


# Register the environment
gym.register(
    id="CAT-v0",
    entry_point="cat_env:CATEnv",
)

if __name__ == "__main__":
    from dataloader import dataset_rl
    from IRT import batch_fisher_information
    from tqdm import tqdm
    from dqn import get_mask
    from cat_wrappers import *
    import time

    train_data, val_data, test_data = dataset_rl()
    # env = gym.make("CAT-v0", dataset=val_data, time_limit_s=60 * 15)  # 15 minutes
    # print(env.observation_space)
    # print(env.observation_space['theta'].sample())
    # print(env.action_space)

    dummy_data = pd.DataFrame(
        data={
            "student_id": [0, 0, 1, 1],
            "item_id": [0, 1, 0, 1],
            "discrimination": [1.0, 1.0, 1.0, 1.0],
            "difficulty": [0.0, 0.0, 0.0, 0.0],
            "response": [1, 1, 1, 1],
            "response_time": [10, 10, 10, 10],
            "final_score": [1, 1, 1, 1],
        }
    )
    dataset = val_data
    n_envs = 4
    def make_env():
        env = gym.make(
            "CAT-v0", 
            dataset=dataset,
            time_limit_s=60 * 15,
            # time_limit_s=15,
        )
        # env = AnsweredVectorObservation(env)  # Async: 105.01 it/s
        # env = DetailedAnsweredVectorObservation(env, dataset=dataset)  # Async: 98.77 it/s
        # env = FisherInfoRewards(env, dataset=dataset)  # Async: 101.76 it/s
        # env = RecordActions(env)  # Async: <5 it/s, Sync: 61.78 it/s
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        # env = MaskObservation(env)
        return env
    start_make = time.time()
    env = gym.vector.AsyncVectorEnv(
        [make_env for _ in range(n_envs)], 
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
        # shared_memory=True,
        copy=False,
    )
    stop_make = time.time()
    print(f"Time taken to make env: {stop_make - start_make}s")  # NOTE: 2.4 seconds for 4 envs. Likely due to sorting the dataset
    # Async: 115.28 it/s (4 envs M2 Mac)
    # Sync: 57.81 it/s (4 envs M2 Mac)

    obs, info = env.reset()
    # print(env.observation_space["main"].shape)

    n = 1000  # number of students
    n_items = dataset["item_id"].unique().shape[0]
    true_thetas = np.zeros(n)
    time_aware = False

    runs_so_far = 0
    start = time.time()
    with tqdm(total=n) as pbar:
        while runs_so_far < n:  # n students
            mask = get_mask((env.num_envs, env.single_action_space.n), info["answered"])
            action = env.action_space.sample(mask=tuple(mask[i] for i in range(mask.shape[0])))

            start_step = time.time()
            observation, reward, terminated, truncated, info = env.step(action)
            stop_step = time.time()
            tqdm.write(f"Step time: {stop_step - start_step}s")
            
            done_mask = np.logical_or(terminated, truncated)
            if done_mask.any():

                start_reset = time.time()
                observation, info = env.reset(options={"mask": done_mask})
                stop_reset = time.time()
                tqdm.write(f"Reset time: {stop_reset - start_reset}s")

                runs_so_far += done_mask.sum()

                pbar.update(done_mask.sum())
    stop = time.time()
    print(f"Time taken: {stop - start}s")
    env.close()
