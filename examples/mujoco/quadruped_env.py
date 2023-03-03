import gym
from gym import spaces
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


class QuadrupedEnv(gym.Env):
    """Every function in this env will return np.ndarray.

    Args:
        gym (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
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
            shape=(self.n_state, ),  # self.n_envs, 
            dtype=float
        )  # low high: TODO

        self.action_space = spaces.Box(low=-10.0, high=10.0, 
                                       shape=(self.n_action, ), dtype=float)  # self.n_envs, 

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.is_async = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return self.n_envs
    
    def _get_info(self, info=None):
        if info:
            info["env_id"] = np.arange(0, self.n_envs)
            return info
        else:
            return {
                "env_id": np.arange(0, self.n_envs)
            }

    def reset_idx(self, env_ids):
        # return: obs_reset, info
        if isinstance(env_ids, np.ndarray):
            env_ids = torch.tensor(env_ids, device=self.device)
        self.env.reset_idx(env_ids)
        env_ids = env_ids.cpu().numpy()
        return self.env.get_observations().cpu().numpy()[env_ids], {"env_id": env_ids}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_state = self.env.reset()[0]  # obs, privileged_obs

        observation = self._agent_state.cpu().numpy()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action, id):
        # action: (self.n_envs, self.n_action)
        
        # print("def step(self, action):", type(action), action.shape)  # <class 'numpy.ndarray'> (4096, 12)
        
        act = torch.zeros(self.n_envs, self.n_action, device=self.device)
        act[id] = torch.tensor(action, device=self.device)
        
        # if len(id) != self.n_envs:
        #     print("id", id)
        # action = torch.tensor(action, device=self.device)
        # print("action.shape", action.shape)
        obs_next, privileged_obs, rew, done, info = self.env.step(act)  # those variables are 4096-dim
        
        observation = obs_next
        info = self._get_info(info)

        # if self.render_mode == "human":
        #     self._render_frame()
        
        num_envs = len(id)  # may less than self.n_envs
        
        # attention: obs_next is torch.tensor, not numpy.ndarray
        # transfer to numpy.ndarray
        obs_next = obs_next.cpu().numpy()
        rew = rew.cpu().numpy()
        done = done.cpu().numpy()  # in QuadrupedEnv, terminated is same as truncated
        for k, v in info["episode"].items():
            # one must make sure that the value of info["episode"][k] is a np.array
            # and the len(self.data) is equal to num_envs
            v_ = v
            if isinstance(v_, torch.Tensor):
                v_ = v_.cpu().numpy()    
                
            if len(v_.shape) == 0:
                v_ = np.array([v_])
            
            if len(v_) != num_envs:
                if len(v_) == 1:
                    info["episode"][k] = np.array([v_.item()] * num_envs)
                elif len(v_) == self.n_envs:
                    info["episode"][k] = v_[id]
                else:
                    raise ValueError("The length of info[episode][k] must be 1 or equal to num_envs")
            else:
                info["episode"][k] = v_
        
        if max(id) >= len(info["time_outs"]):
            print("len(id), id", len(id), id)
            print("len(info[\"time_outs\"])", len(info["time_outs"]))
            # 为什么time_outs的长度会小于4096呢？
            new_id = []
            for x in id:
                if x < len(info["time_outs"]):
                    new_id.append(x)
            id = new_id
            print("new_id!!!")
        
        if isinstance(info["time_outs"], torch.Tensor):
            info["time_outs"] = info["time_outs"][id].cpu().numpy()
        elif isinstance(info["time_outs"], np.ndarray):
            info["time_outs"] = info["time_outs"][id]  # FIXME: IndexError: index 4094 is out of bounds for axis 0 with size 4094
        # 注意一下info["time_outs"]里面对应的是4096中的哪些维度，id指的应该是原先的4096中的那些维度
            
        for k, v in info.items():
            if k != "episode" and k != "time_outs":
                if isinstance(v, torch.Tensor):
                    info[k] = v[id].cpu().numpy()
                elif isinstance(v, np.ndarray):
                    info[k] = v[id]
                else:
                    raise ValueError("The type of info[k] must be torch.Tensor or np.ndarray")
        
        return obs_next[id], rew[id], done[id], done[id], info

    def render(self):
        pass
    
    def close(self):
        pass
