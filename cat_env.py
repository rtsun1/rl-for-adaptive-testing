import gymnasium as gym
import numpy as np
from typing import Optional
import pandas as pd

from IRT import calculate_eap_ability


class CATEnv(gym.Env):
    def __init__(
        self,
        dataset_f: str,
        time_limit_s: float = 20.0,
        init_theta: float = 0.0,
        theta_lb: float = -4.0,
        theta_ub: float = 4.0,
    ):
        """
        Define the observation and action spaces

        Observation space: {"theta": float, "time_left": float}

        Action space: [0, MAX_Q]

        dataset_f: path to the dataset file
        time_limit_s: max time for a test in seconds
        init_theta: initial ability estimate
        theta_lb: lower bound for theta
        theta_ub: upper bound for theta
        """
        super().__init__()  # may not be necessary, but it's good practice

        self.dataset = self._load_dataset(dataset_f)
        self.time_taken_s = 0
        self.time_limit_s = time_limit_s
        self.theta = init_theta
        self.theta_range = (theta_lb, theta_ub)

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
        n_items = self.dataset["item_id"].unique().shape[0]
        self.action_space = gym.spaces.Discrete(n_items)

        self._action_to_idx = {
            action: id for action, id in enumerate(self.dataset["item_id"].unique())
        }

    def _load_dataset(self, dataset_f: str) -> pd.DataFrame:
        """
        Load the dataset

        return: DataFrame sorted by student_id and item_id
            cols: ["student_id", "item_id", "discrimination", "difficulty", "response", "response_time", "final_score"]
        """
        raw = pd.read_csv(dataset_f)
        raw = raw.sort_values(by=["student_id", "item_id"])
        raw = raw.reset_index(drop=True)

        n_items = raw["item_id"].unique().shape[0]
        if raw.shape[0] % n_items != 0:
            raise ValueError("Each student must have the same number of items")

        return raw

    def _get_obs(self):
        """
        Get the current observation
        """
        return {
            "theta": np.array(self.theta, dtype=np.float32).reshape(
                1
            ),  # NOTE: reshaping to (1,) from () to make it consistent with the Box space
            "time_left": np.array(
                self.time_limit_s - self.time_taken_s, dtype=np.float32
            ).reshape(1),
        }

    def _get_info(self):
        """
        Get the current info
        """
        return {"student_id": self.sid}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Initialize a new episode

        Start state is no response and time taken is 0.
        Use seed to select the student.
        Options currently not used.
        """
        super().reset(seed=seed)  # For seeding self.np_random

        # Select a student
        self.sid = self.np_random.choice(self.dataset["student_id"].unique())

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
        if action in self.answered:
            return (
                self._get_obs(),
                -1,
                False,
                False,
                self._get_info(),
            )  # NOTE: returning a reward of -1. might need to tune based on the reward function
        # NOTE: Another potential problem above is an infinite loop if the agent keeps selecting the same item (if this reward is better than normal rewards)

        # Otherwise, get the item data
        item = self._action_to_idx[action]
        item_data = self.dataset[self.dataset["student_id"] == self.sid]
        item_data = item_data[item_data["item_id"] == item]
        if item_data.empty:
            raise ValueError(
                f"Item ID {item} not found in the dataset. For student ID {self.sid}"
            )

        # Extract the item parameters
        discrimination = item_data["discrimination"].values[0]
        difficulty = item_data["difficulty"].values[0]
        response = item_data["response"].values[0]
        response_time = item_data["response_time"].values[0]

        # If the student couldn't answer the question before time ran out, terminate the episode
        self.time_taken_s += response_time
        if self.time_taken_s > self.time_limit_s:
            all_responses = self.dataset[self.dataset["student_id"] == self.sid][
                "response"
            ].values  # NOTE: could optimize speed by storing the sid's row range if data is sorted
            all_discrimination = self.dataset[self.dataset["student_id"] == self.sid][
                "discrimination"
            ].values
            all_difficulty = self.dataset[self.dataset["student_id"] == self.sid][
                "difficulty"
            ].values
            true_theta, _ = calculate_eap_ability(
                all_responses, all_discrimination, all_difficulty, self.theta_range
            )

            reward = -(
                (true_theta - self.theta) ** 2
            )  # NOTE: reward is the negative of the squared error in the ability estimate
            return (
                self._get_obs(),
                reward,
                True,
                False,
                self._get_info(),
            )  # return the error in the ability estimate (from the entire dataset vs the student's responses)

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
    env = gym.make("CAT-v0", dataset_f="data/test_data.csv")
    print(env.observation_space)
    print(env.action_space)
    observation, info = env.reset()
    print(observation, info)

    episode_over = False
    G = 0
    while not episode_over:
        action = (
            env.action_space.sample()
        )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        G += reward
        print(observation, reward, info)
        episode_over = terminated or truncated

    env.close()
