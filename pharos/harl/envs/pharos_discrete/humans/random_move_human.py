from typing import Optional, override
import numpy as np
from harl.envs.pharos_discrete.humans.base import Human


class RandomMoveHuman(Human):
    def __init__(
        self,
        id: str,
        scare_factor: float,
        mean_speed: float,
        std_speed: float,
        position: np.ndarray = None,
        velocity: Optional[np.ndarray] = None,
    ):
        self.human_speed_distribution = np.random.normal(mean_speed, std_speed)
        if velocity is None:
            velocity = self._gen_random_2d_speed()
            velocity = np.array([velocity[0], 0, velocity[1]])
        super().__init__(id, scare_factor, position, velocity)
        self.change_velocity_interval = 10000  # 10s
        self.timestamp = 0

    @override
    def update(self, delta_t: float) -> None:
        """
        更新人类的状态, delta_t: s
        """
        self.position += self.velocity * delta_t
        self.timestamp += delta_t
        # 如果达到间隔则重新生成速度
        if self.timestamp >= self.change_velocity_interval:
            self.timestamp = 0
            # 随机生成一个新速度
            self.velocity = self._gen_random_2d_speed()
            self.velocity = np.array([self.velocity[0], 0, self.velocity[1]])

    def _gen_random_2d_speed(self) -> np.ndarray:
        """
        随机生成一个2D速度
        :return: 2D速度
        """
        # 随机生成一个方向
        angle = np.random.uniform(0, 2 * np.pi)
        # 计算新的速度
        new_velocity = self.human_speed_distribution * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        return new_velocity
