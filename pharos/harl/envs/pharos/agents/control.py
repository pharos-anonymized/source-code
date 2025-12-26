from abc import ABC, abstractmethod

import numpy as np

from harl.envs.pharos.utils.utils import SafeSpaceParams
class Agent:
    def __init__(self, id: int, name:str, weight=10, max_speed=10, max_acceleration=10, pos=(0, 0, 0),
                 init_vertex=(0, 0, 0), observation_agents=5, delta_t=0.1, target_pos=(0, 0, 0)):
        self.id = id
        self.name = name
        self.weight = weight
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.pos = pos
        self.init_pos = pos
        self.vertex = init_vertex
        self.observation = np.zeros(shape=(observation_agents + 1, 6)) # x,y,z,vx,vy,vz
        self.delta_t = delta_t
        self.target_pos = target_pos
        self.time_consumed = 0
        self.safe_space_aabb_min_abs = (0, 0, 0) # 绝对坐标
        self.safe_space_aabb_max_abs = (0, 0, 0)
        self.moving = True

    def set_target_pos(self, target_pos):
        self.target_pos = target_pos

    def __str__(self):
        return f"Agent {self.name} with weight {self.weight}, max speed {self.max_speed}, and max acceleration {self.max_acceleration}"

    def __repr__(self):
        return f"Agent({self.name}, {self.weight}, {self.max_speed}, {self.max_acceleration})"

    @abstractmethod
    def estimate(self, save_space: SafeSpaceParams) -> np.ndarray:
        """
        Called every delta_t
        Args:
            save_space: safe space for the agent now
        Returns:
            np.ndarray: 预测状态, delta_t之后的坐标+速度 ((x, y, z), (vx, vy, vz))
        """
        pass


    def update(self, real_pos: np.ndarray, real_vertex: np.ndarray) -> None:
        """
        Called every delta_t
        Args:
            save_space: safe space for the agent now
        """
        self.pos = real_pos
        self.vertex = real_vertex

    def state(self) -> np.ndarray:
        """
        返回当前agent的state
        """
        return np.array(self.pos, self.vertex)
    
    def reset(self, pos: np.ndarray, vertex: np.ndarray, target_pos: np.ndarray) -> None:
        """
        重置agent的状态
        """
        self.pos = pos
        self.vertex = vertex
        self.target_pos = target_pos