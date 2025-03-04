import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from gymnasium.spaces import Box, Discrete

from cat_env import CATEnv

class AnsweredVectorObservation(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_items = env.env.action_space.n
        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "theta": env.observation_space["theta"],
                "time_left": env.observation_space["time_left"],
                "answered": gym.spaces.MultiBinary(n=self.n_items),
            }
        )
    
    def _add_answered(self, obs, answered):
        answered = list(answered)  # set to list
        answered_arr = np.zeros(shape=self.n_items, dtype=np.bool)
        answered_arr[answered] = True
        obs["answered"] = answered_arr
        return obs
    
    def reset(self, *, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        return self._add_answered(obs, info["answered"]), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_answered(obs, info["answered"]), reward, terminated, truncated, info
    
if __name__ == "__main__":
    env = gym.make("CAT-v0", dataset_f="data/test_data.csv", time_limit_s=20.0)
    env = AnsweredVectorObservation(env)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(1)
    obs, reward, terminated, truncated, info = env.step(2)
    obs, reward, terminated, truncated, info = env.step(5)
    print(obs)