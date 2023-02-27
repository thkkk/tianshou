import gym
from gym import spaces
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


class QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, task="aliengo_stand", num_envs=4096, render_mode=None):
        args = get_args()
        if render_mode is None:
            args.headless = True
        args.task = task
        args.num_envs = num_envs
        self.args = args
        self.env, self.env_cfg = task_registry.make_env(name=args.task, args=args)
        
        self.n_envs = self.env_cfg.env.num_envs
        self.n_state = self.env_cfg.env.num_observations
        self.n_action = self.env_cfg.env.num_actions

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(self.n_envs, self.n_state), 
            dtype=float
        ),  # low high: TODO

        self.action_space = spaces.Box(low=-10.0, high=10.0, 
                                       shape=(self.n_envs, self.n_action), dtype=float)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def __len__(self):
        return self.n_envs
    
    def _get_info(self):
        return {
            "env_id": np.arange(0, self.n_envs)
        }

    def reset_idx(self, env_ids):
        # return: obs_reset, info
        # self.env.reset_idx(env_ids)
        pass
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_state = self.env.reset()[0]

        observation = self._agent_state
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        # action: (self.n_envs, self.n_action)
        
        state_next, privileged_obs, reward, done, info = self.env.step(action)  # FIXME: cuda out of memory
        
        observation = state_next
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return state_next, reward, done, done, info

    def render(self):
        pass
    
    def close(self):
        pass