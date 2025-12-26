from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from harl.envs.pharos_discrete.config import SafeSpaceParams, SafeSpaceParamsContinuous


class Agent:
    def __init__(
        self,
        id: int,
        name: str,
        target_pos: np.ndarray[(3,), int],
        max_speed_cube: int,
        init_pos: np.ndarray[(3,), int],
        init_velocity: np.ndarray[(3,), int] = np.zeros(3),
    ):
        self.id = id
        self.name = name
        self.target_pos = target_pos
        self.pos = init_pos
        self.velocity = init_velocity
        self.max_speed_cube = max_speed_cube
        self.arrived = False

    def set_target_pos(self, target_pos: np.ndarray[(3,), int]):
        self.target_pos = target_pos

    def __str__(self):
        return f"Agent {self.name}"

    def __repr__(self):
        return f"Agent(id: {self.id}, name: {self.name})"

    @abstractmethod
    def estimate(
        self, save_space: SafeSpaceParams | SafeSpaceParamsContinuous
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Called every delta_t
        Args:
            save_space: safe space for the agent now
            delta_t: time step for the estimation
        Returns:
            np.ndarray: 预测状态, delta_t之后的坐标+速度 ((x, y, z), (vx, vy, vz))
        """
        pass

    def update(
        self, real_pos: np.ndarray[(3,), int], real_velocity: np.ndarray[(3,), int]
    ) -> None:
        """
        Called every delta_t
        Args:
            save_space: safe space for the agent now
        """
        self.pos = real_pos
        self.velocity = real_velocity
        if np.linalg.norm(real_pos - self.target_pos) < 1e-2:
            self.arrived = True

    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回当前agent的state
        """
        return self.pos, self.velocity
