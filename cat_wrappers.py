import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.spaces import Box, Discrete

from cat_env import CATEnv
from IRT import batch_fisher_information, calculate_eap_ability
from ability_level_model import RegressionModel
import torch

class FisherInfoRewards(Wrapper):
    """
    Reward wrapper that returns the Fisher Information of the student's ability estimate
    divided by the response time.
    """

    def __init__(self, env, dataset, reward_std=0.1182895876004875, reward_mean=0.17804979690015083):  # these are the stats of FI at theta=0 for train set
        super().__init__(env)
        self.reward_std = reward_std
        self.reward_mean = reward_mean
        self.answered = set()  # a reference to CATEnv's answered so we can check it before calling step
        self.dataset = dataset
        self.student = None
        self.theta = None

    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        self.answered = info["answered"]
        self.student = info["sid_idx"] * self.action_space.n
        self.theta = float(info["theta"])  # float to ensure this is not a ref

        return obs, info
    
    def step(self, action):
        # Check if the question has already been answered
        already_answered = action in self.answered
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.answered = info["answered"]

        if terminated or truncated:
            reward = 0  # no reward for terminal states
        if terminated or truncated or already_answered:
            return obs, reward, terminated, truncated, info
        
        # Extract relevant information
        difficulty = self.dataset.iloc[action]["difficulty"]
        discrimination = self.dataset.iloc[action]["discrimination"]
        # response_time = self.dataset.iloc[self.student + action]["response_time"]

        # Calculate the fisher information
        fisher_info = batch_fisher_information(difficulty, discrimination, self.theta)
        self.theta = float(info["theta"])
        reward = (fisher_info - self.reward_mean) / self.reward_std

        return obs, reward, terminated, truncated, info

class SparseMSERewards(Wrapper):
    """
    Reward wrapper that returns the Fisher Information of the student's ability estimate
    divided by the response time.
    """
    def __init__(self, env, dataset, reward_std=0.1182895876004875, reward_mean=0.17804979690015083):  # these are the stats of FI at theta=0 for train set
        super().__init__(env)
        self.reward_std = reward_std
        self.reward_mean = reward_mean
        self.answered = set()  # a reference to CATEnv's answered so we can check it before calling step
        self.dataset = dataset
        self.student = None
        self.theta = None
        self.MSE_last = None

    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        self.theta = float(info["theta"])  # float to ensure this is not a ref
        self.MSE_last = 0.0

        return obs, info
    
    def step(self, action):
        # Check if the question has already been answered
        already_answered = action in self.answered
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated or already_answered:
            reward = 0  # no reward for terminal states
        if terminated or truncated or already_answered:
            return obs, reward, terminated, truncated, info
        
        # Calculate Squared Error
        self.theta = float(info["theta"])
        self.true_theta = float(info["true_theta"])
        MSE_curr = (self.theta - self.true_theta) ** 2

        reward = (self.MSE_last - MSE_curr) * 10
        self.MSE_last = MSE_curr

        return obs, reward, terminated, truncated, info

class MLThetaObservationAndInfo(Wrapper):
    def __init__(self, env, dataset, device="cpu", checkpoint="ml_weights/best_model.pt"):
        super().__init__(env)
        self.reg_model = RegressionModel().to(device)
        self.reg_model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        self.dataset = dataset
        self.n_items = env.action_space.n
        self.device = device
    
    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)

        self.sid_idx = info["sid_idx"]
        start = self.sid_idx * self.n_items
        stop = (self.sid_idx + 1) * self.n_items
        self.torch_dataset = self.dataset[start:stop]
        self.torch_dataset = torch.tensor(self.torch_dataset[["discrimination", "difficulty", "response", "response_time"]].values).float().to(self.device)

        return obs, info  # leave default theta at reset
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        answered = info["answered"]  # include the question just asked for the model

        curr_data = self.torch_dataset
        curr_state = torch.ones_like(curr_data) * -1
        curr_state[list(answered)] = curr_data[list(answered)]
        curr_state = curr_state.reshape((1, curr_state.shape[0], curr_state.shape[1]))
        estimated_ability_level = self.reg_model(curr_state)[0].item()

        obs["theta"] = estimated_ability_level
        info["theta"] = estimated_ability_level

        return obs, reward, terminated, truncated, info
    
class WeightedMLThetaObservationAndInfo(Wrapper):
    def __init__(self, env, dataset, device="cpu"):
        super().__init__(env)
        self.reg_model = RegressionModel().to(device)
        self.reg_model.load_state_dict(torch.load("ml_weights/best_model.pt", weights_only=True, map_location=device))
        self.dataset = dataset
        self.n_items = env.action_space.n
        self.device = device
    
    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)

        self.sid_idx = info["sid_idx"]
        start = self.sid_idx * self.n_items
        stop = (self.sid_idx + 1) * self.n_items
        self.torch_dataset = self.dataset[start:stop]
        self.torch_dataset = torch.tensor(self.torch_dataset[["discrimination", "difficulty", "response", "response_time"]].values).float().to(self.device)

        return obs, info  # leave default theta at reset
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        answered = info["answered"]  # include the question just asked for the model

        curr_data = self.torch_dataset
        curr_state = torch.ones_like(curr_data) * -1
        curr_state[list(answered)] = curr_data[list(answered)]
        curr_state = curr_state.reshape((1, curr_state.shape[0], curr_state.shape[1]))
        estimated_ability_level = self.reg_model(curr_state)[0].item()

        weight_func = (20 + len(list(answered))) / 90
        weight_model = 1 - weight_func
        theta = weight_model * estimated_ability_level + weight_func * obs["theta"].item()

        obs["theta"] = theta
        info["theta"] = theta

        return obs, reward, terminated, truncated, info

class AnsweredVectorObservation(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_items = env.action_space.n
        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "answered": gym.spaces.MultiBinary(n=self.n_items),
                **env.observation_space
            }
        )
    
    def _add_answered(self, obs, answered):
        answered = list(answered)  # set to list
        answered_arr = np.zeros(shape=self.n_items, dtype=bool)
        answered_arr[answered] = True
        obs["answered"] = answered_arr
        return obs
    
    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        return self._add_answered(obs, info["answered"]), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_answered(obs, info["answered"]), reward, terminated, truncated, info
    
class MaskObservation(AnsweredVectorObservation):
    def __init__(self, env):
        super(AnsweredVectorObservation, self).__init__(env)
        self.n_items = env.action_space.n
        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "answered": gym.spaces.MultiDiscrete(nvec=[2] * self.n_items, dtype=np.int8),
                "main": env.observation_space
            }
        )
    
    def _add_answered(self, obs, answered):
        answered = list(answered)  # set to list
        answered_arr = np.zeros(shape=self.n_items, dtype=np.int8)
        answered_arr[answered] = 1
        return {
            "main": obs,
            "answered": answered_arr
        }
    
class DetailedAnsweredVectorObservation(Wrapper):
    """
    Add to the observation space an array (n_items, 4)
    with columns: discrimination, difficulty, response, response time.
    -1 in all entries for rows corresponding to unanswered questions.
    """
    def __init__(self, env, dataset):
        super().__init__(env)
        self.details_arr = -np.ones(shape=(env.action_space.n, 4), dtype=np.float32)
        self.dataset = dataset  # a reference to CATEnv's dataset for filling in details
        self.answered = set()  # a reference to CATEnv's answered so we can check it before calling step
        self.student = None  # the starting index of the student's questions in the dataset

        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "detailed_answered": gym.spaces.Box(
                    low=-1,  # difficulty could be between (-inf, inf) but likely lower bounded by -1
                    high=np.inf,  # NOTE: for now, no upper bound on any of the columns (only impacts observation_space.sample())
                    shape=(env.action_space.n, 4), 
                    dtype=np.float32
                ),
                **env.observation_space
            }
        )
    
    def _add_answered(self, obs):
        details_arr = self.details_arr.view()
        details_arr.flags.writeable = False  # don't let user mess with wrapper's details_arr
        obs["detailed_answered"] = details_arr
        return obs
    
    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        self.details_arr.fill(-1)
        self.answered = info["answered"]
        self.student = info["sid_idx"] * self.action_space.n

        return self._add_answered(obs), info
    
    def step(self, action):
        if action not in self.answered:
            # fill in details_arr
            self.details_arr[action, 0] = self.dataset.iloc[action]["discrimination"]  # assuming sorted by student_id then question_id
            self.details_arr[action, 1] = self.dataset.iloc[action]["difficulty"]
            self.details_arr[action, 2] = self.dataset.iloc[self.student + action]["response"]
            self.details_arr[action, 3] = self.dataset.iloc[self.student + action]["response_time"]

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.answered = info["answered"]  # incase info["answered"] is not a reference

        return self._add_answered(obs), reward, terminated, truncated, info
    

class UncertaintyVectorObservation(Wrapper):
    """
    Add to the observation space a scalar for the standard error 
    in the ability estimate calculated by IRT function
    """
    def __init__(self, env, dataset, theta_lb=-4.0, theta_ub=4.0):
        super().__init__(env)
        self.dataset = dataset
        self.answered = set()
        self.student = None
        
        # To calculate ability estimate and uncertainty
        self.theta_range = (theta_lb, theta_ub)
        self.discrimination = []
        self.difficulty = []
        self.response = []

        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "uncertainty": gym.spaces.Box(
                    low=0, # 0 standard error is the lower limit
                    high=np.inf, # Standard error -> inf theoretically
                    shape=(1,),
                    dtype=np.float32
                ), 
                **env.observation_space
            }
        )

    def _add_uncertainty(self, obs):
        _, uncertainty = calculate_eap_ability(self.response, self.discrimination, self.difficulty, self.theta_range)
        obs["uncertainty"] = uncertainty
        return obs

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.answered = info["answered"]
        self.student = info["sid_idx"] * self.action_space.n

        self.discrimination = []
        self.difficulty = []
        self.response = []

        return self._add_uncertainty(obs), info
    
    def step(self, action):
        if action not in self.answered:
            # Fill in the discrimination, difficulty and resposnse
            self.discrimination.append(self.dataset.iloc[action]["discrimination"])
            self.difficulty.append(self.dataset.iloc[action]["difficulty"])
            self.response.append(self.dataset.iloc[self.student + action]["response"])

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.answered = info["answered"]

        return self._add_uncertainty(obs), reward, terminated, truncated, info

    
class RecordActions(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = []  # a list for each episode
        self.answered = set()  # a reference to CATEnv's answered so we can check it before calling step
    
    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        self.actions.append([])
        self.answered = info["answered"]
        info["action_history"] = self.actions
        return obs, info
    
    def step(self, action):
        if action not in self.answered:
            self.actions[-1].append(action)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.answered = info["answered"]  # incase info["answered"] is not a reference

        info["action_history"] = self.actions
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    from dataloader import dataset_rl
    train, val, test = dataset_rl()

    # # FisherInfoRewards
    # env = gym.make("CAT-v0", dataset=val)
    # env = FisherInfoRewards(env, dataset=val)
    # obs, info = env.reset()
    # obs, reward, terminated, truncated, info = env.step(1)
    # assert reward > 0
    # obs, reward, terminated, truncated, info = env.step(2)
    # assert reward > 0
    # obs, reward, terminated, truncated, info = env.step(5)
    # assert reward > 0
    # obs, reward, terminated, truncated, info = env.step(1)
    # assert reward == -1

    # while not terminated:  # finish the episode
    #     a = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(a)
    # assert reward < 0  # should be -(squared error)
    # env.close()

    # # AnsweredVectorObservation
    # env = gym.make("CAT-v0", dataset=val)
    # env = AnsweredVectorObservation(env)
    # # env = gym.vector.SyncVectorEnv([lambda: env])
    # obs, info = env.reset()
    # obs, reward, terminated, truncated, info = env.step(1)
    # obs, reward, terminated, truncated, info = env.step(2)
    # obs, reward, terminated, truncated, info = env.step(5)
    # env.close()
    # print(obs)

    # # MaskObservation
    # env = gym.make("CAT-v0", dataset=val)
    # env = MaskObservation(env)
    # obs, info = env.reset()
    # obs, reward, terminated, truncated, info = env.step(1)
    # obs, reward, terminated, truncated, info = env.step(2)
    # obs, reward, terminated, truncated, info = env.step(5)
    # print(obs)
    # obs, info = env.reset()
    # print(obs)
    # obs, reward, terminated, truncated, info = env.step(1)
    # print(obs)
    # env.close()
    

    # # DetailedAnsweredVectorObservation
    # env = gym.make("CAT-v0", dataset=val)
    # env = DetailedAnsweredVectorObservation(env, dataset=val)
    # obs, info = env.reset()
    # obs, reward, terminated, truncated, info = env.step(1)
    # obs, reward, terminated, truncated, info = env.step(2)
    # obs, reward, terminated, truncated, info = env.step(5)
    # obs, reward, terminated, truncated, info = env.step(1)
    # obs, reward, terminated, truncated, info = env.step(10)
    # print(obs)
    # env.reset()
    # obs, reward, terminated, truncated, info = env.step(1)
    # obs, reward, terminated, truncated, info = env.step(1)
    # print(obs)
    # env.close()

    # UncertaintyVectorObservation
    env = gym.make("CAT-v0", dataset=val)
    env = UncertaintyVectorObservation(env, dataset=val)
    obs, info = env.reset()
    print(obs)
    obs, reward, terminated, truncated, info = env.step(1)
    print(obs)
    obs, reward, terminated, truncated, info = env.step(2)
    print(obs)
    obs, reward, terminated, truncated, info = env.step(5)
    print(obs)
    obs, reward, terminated, truncated, info = env.step(1)
    print(obs)
    obs, reward, terminated, truncated, info = env.step(10)
    print(obs)
    env.close()


    # # RecordActions
    # env = gym.make("CAT-v0", dataset=val)
    # env = RecordActions(env)
    # obs, info = env.reset()
    # print(info["action_history"])
    # obs, reward, terminated, truncated, info = env.step(1)
    # print(info["action_history"])
    # obs, reward, terminated, truncated, info = env.step(2)
    # obs, reward, terminated, truncated, info = env.step(5)
    # env.reset()
    # obs, reward, terminated, truncated, info = env.step(1)
    # obs, reward, terminated, truncated, info = env.step(1)
    # obs, reward, terminated, truncated, info = env.step(2)

    # env.close()
    # print(info["action_history"])