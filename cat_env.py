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
        raw = raw.sort_values(by=["student_id", "item_id"])  # so we can access the items in O(1) time
        raw = raw.reset_index(drop=True)

        n_items = raw["item_id"].unique().shape[0]

        # NOTE: Assumptions for O(1) lookup: 1. Items are sorted by student_id and item_id, 2. Each student has the same number of items
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
        return {
            "student_id": self.sid, 
            "dataset": self.dataset,
            "conversion": self._action_to_idx,
            "theta": self.theta,
            "true_theta": self.true_theta,
            "average_times": self.average_times,
            "answered": self.answered,
            "sid_idx": self.sid_idx,
        }
    
    def _get_true_theta(self):
        """
        Get the true theta for the student
        """
        start = self.sid_idx * self.n_items
        stop = (self.sid_idx + 1) * self.n_items

        all_responses = self.dataset.iloc[start:stop][
            "response"
        ].values
        all_discrimination = self.dataset[start:stop][
            "discrimination"
        ].values
        all_difficulty = self.dataset[start:stop][
            "difficulty"
        ].values

        true_theta, _ = calculate_eap_ability(
            all_responses, all_discrimination, all_difficulty, self.theta_range
        )

        return true_theta
        

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
        self.true_theta = self._get_true_theta()
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
            return (
                self._get_obs(),
                -1,
                False,
                False,
                self._get_info(),
            )  # NOTE: returning a reward of -1. might need to tune based on the reward function
        # NOTE: Another potential problem above is an infinite loop if the agent keeps selecting the same item (if this reward is better than normal rewards)

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
            reward = 50 * - (self.true_theta - self.theta) ** 2  # NOTE: reward is the negative of the squared error in the ability estimate
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
    import time
    from dqn import get_mask

    train_data, val_data, test_data = dataset_rl()
    env = gym.make("CAT-v0", dataset=val_data, time_limit_s=60 * 15)  # 15 minutes
    print(env.observation_space)
    print(env.observation_space['theta'].sample())
    print(env.action_space)

    # Greedy FI Algorithm
    obs, info = env.reset()

    n = 1  # number of students
    n_items = info["dataset"]["item_id"].unique().shape[0]
    diffs = info["dataset"]["difficulty"].unique()  # unique preserves order of appearance
    discs = info["dataset"]["discrimination"].unique()
    thetas = np.zeros(n)  # estimated thetas
    true_thetas = np.zeros(n)
    time_aware = False

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

    # Theta RMSE:
    print("Estimated Thetas:")
    print(thetas.round(2))
    print("True Thetas:")
    print(true_thetas.round(2))
    rmse = np.sqrt(np.mean((thetas - true_thetas) ** 2))
    
    print("RMSE: ", rmse)

    env.close()
