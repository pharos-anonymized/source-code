from typing import Tuple
from typing import override
import numpy as np

from harl.envs.pharos_discrete.agents.control import Agent
from harl.envs.pharos_discrete.config import SafeSpaceParams, SafeSpaceParamsContinuous
from harl.envs.pharos_discrete.utils import ray_aabb_intersection


class NaiveControlAgent(Agent):
    @override
    def estimate(
        self, safe_space: SafeSpaceParams | SafeSpaceParamsContinuous
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(safe_space, SafeSpaceParams):
            move_cubes = safe_space.to_space()
            move_cubes = np.minimum(
                move_cubes,
                np.ones(
                    3,
                )
                * self.max_speed_cube,
            )
            return self.pos + move_cubes, move_cubes
        else:
            # 按当前位置到连续空间中心指向的方向移动
            box = safe_space.to_space(moveable=True)
            center = (box[0] + box[1]) / 2
            direction = center - self.pos
            if np.linalg.norm(direction) < 1e-2:
                return self.pos, np.zeros(3)
            direction = direction / np.linalg.norm(direction)
            bound = ray_aabb_intersection(
                box[0] + self.pos, box[1] + self.pos, self.pos, direction
            )
            if bound is None:
                warning_msg = f"Agent {self.name} isn't in safe_space at {self.pos}"
                print(warning_msg)
                return self.pos, np.zeros(3)
            move_distance = min(
                np.linalg.norm(direction) * self.max_speed_cube,
                np.linalg.norm(bound - self.pos),
            )
            return self.pos + direction * move_distance, direction * move_distance
