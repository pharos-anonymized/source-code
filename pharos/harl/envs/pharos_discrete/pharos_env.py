import os
import numpy as np
from gym import spaces
from typing import Tuple, List, Dict, Any
from harl.envs.pharos_discrete.env_core import EnvCore
from harl.envs.pharos_discrete.config import MoveActionNum
import warnings

from termcolor import colored


class PharosDiscreteEnv:
    def __init__(self, args):
        # 初始化底层环境
        print(colored(args, "red"))
        for idx, arg in enumerate(args):
            if isinstance(arg, str) and len(arg.split(",")) > 1:
                args[idx] = arg.split(",")  # 实际传递的是list
                print(colored(args[idx], "red"))
        self.env = EnvCore(args)

        # 设置智能体数量
        self.n_agents = self.env.agent_num

        # 定义观察空间和动作空间
        # 每个智能体的观察空间
        obs_dim = self.env.obs_dim
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        # 共享观察空间, 即state
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.env.state_dim,), dtype=np.float32
            )
            for _ in range(self.n_agents)
        ]
        # 动作空间
        # self.action_space = [
        #     spaces.MultiDiscrete([MoveActionNum, Agent_Max_Step_Cube])
        #     for _ in range(self.n_agents)
        # ]
        if self.env.tp.action_discrete:
            self.action_space = [
                spaces.Discrete(MoveActionNum) for _ in range(self.n_agents)
            ]
        else:
            # 这里Box不是实际约束, 只是采样的时候会这样采
            self.action_space = [
                spaces.Box(
                    low=0, high=np.inf, shape=(self.env.action_dim,), dtype=np.float32
                )
                for _ in range(self.n_agents)
            ]

    def step(
        self, actions: List[np.ndarray]
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[float],
        List[bool],
        Dict[str, Any],
        List[np.ndarray] | None,
    ]:
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        tuple[list, list, list[list], list, list, list]
        """
        # 执行环境步进
        results = self.env.step(actions)
        obs, rewards, dones, infos, available_actions = results

        # 构造共享状态
        state = self.env.get_state()
        states = [state for _ in range(self.n_agents)]
        return obs, states, rewards, dones, infos, available_actions

    def reset(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray] | None]:
        # 重置环境
        obs, states, available_actions = self.env.reset()

        return obs, states, available_actions

    def seed(self, seed: int) -> None:
        """设置随机种子"""
        self.env.seed(seed)

    def close(self) -> None:
        """关闭环境"""
        self.env.close()

    def enable_trace(self):
        # print(f"PharosEnv: enabling trace in process {os.getpid()}")
        self.env.enable_trace()
        # print(f"PharosEnv: trace enabled, env.tracing = {self.env.tracing}")

    def disable_trace(self):
        # print("Disable tracing")
        self.env.disable_trace()

    def reset_trace(self):
        # print("Reset tracing")
        self.env.reset_trace()

    def get_tracing_info(self) -> Dict[str, Any]:
        # print("discrete env get_tracing_info")
        return self.env.get_tracing_info()
