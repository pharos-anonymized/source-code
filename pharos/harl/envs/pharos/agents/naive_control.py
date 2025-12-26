import numpy as np
from harl.envs.pharos.agents.control import Agent
from typing import override
from harl.envs.pharos.utils.utils import (
    normalize,
    ray_aabb_intersection,
    SafeSpaceParams,
)


class NaiveControlAgent(Agent):
    def __init__(
        self,
        id: int,
        name: str,
        weight=10,
        max_speed=10,
        max_acceleration=10,
        pos=(0, 0, 0),
        init_vertex=(0, 0, 0),
        observation_agents=5,
        delta_t=0.1,
        target_pos=(0, 0, 0),
        Cube_Min_Half_Length=0.25,  # 安全空间的最小半边长(飞机自身特征长度的一半)
    ):
        super().__init__(
            id,
            name,
            weight,
            max_speed,
            max_acceleration,
            pos,
            init_vertex,
            observation_agents,
            delta_t,
            target_pos,
        )
        self.Cube_Min_Half_Length = (
            Cube_Min_Half_Length  # 安全空间的最小半边长(飞机自身特征长度的一半)
        )

    @override
    def estimate(self, save_space: SafeSpaceParams) -> np.ndarray:
        """
        根据坐标点和当前位置直接做出一条直线, 直接飞到与安全空间的交点(如果速度允许的话)
        ATTENTION: 重新计算速度, 但不更新, 只是estimate
        """
        aabb_min, aabb_max = save_space.to_AABB()  # 注意是相对坐标
        aabb_min += np.ones_like(aabb_min) * self.Cube_Min_Half_Length
        aabb_max -= np.ones_like(aabb_max) * self.Cube_Min_Half_Length
        center = (aabb_min + aabb_max) / 2
        safe_space_direction = normalize(center)

        # 计算的是中心点的可移动范围
        # 用于渲染
        safe_space_aabb_max_abs = self.pos + aabb_max
        safe_space_aabb_min_abs = self.pos + aabb_min  # 注意是绝对坐标
        if (
            np.linalg.norm(self.target_pos - self.pos) / self.max_speed < self.delta_t
            and np.all(self.target_pos >= safe_space_aabb_min_abs)
            and np.all(self.target_pos <= safe_space_aabb_max_abs)
        ):
            # 说明在安全空间内且可以直接到达
            # print(f"agent {self.name}: {self.init_pos} --> {self.target_pos}, {self.time_consumed} s")
            self.moving = False
            return np.array([self.target_pos, np.zeros(3)])
        # ray_direction = normalize(np.array(self.target_pos) - np.array(self.pos))
        ray_direction = safe_space_direction
        intersection_pos = ray_aabb_intersection(
            aabb_min + self.pos, aabb_max + self.pos, self.pos, ray_direction
        )

        assert intersection_pos is not None
        estimate_time = np.linalg.norm(intersection_pos - self.pos) / self.max_speed
        # 能最大就最大, 不能取在安全范围内的最大
        if estimate_time > self.delta_t:
            pos = self.pos + self.max_speed * ray_direction * self.delta_t
            return np.array([pos, self.max_speed * ray_direction])
        else:
            pos = intersection_pos
            return np.array([pos, (intersection_pos - self.pos) / self.delta_t])
