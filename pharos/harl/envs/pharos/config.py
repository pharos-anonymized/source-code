import warnings
from dataclasses import dataclass, field
from typing import Tuple
from enum import Enum
import numpy as np
import torch


@dataclass
class RewardParams:
    closer_factor: float  # 速度奖励系数
    collision_factor: float  # 重叠惩罚系数, >= 1 * reach finally
    reach_factor: float  # 到达目标的奖励系数


@dataclass
class TrainingParam:
    # ATTENTION: 连续时, 无cube_len, 空间参数为m/s, 时间参数为ms
    stage_name: str = "stage1"  # 训练阶段名称, 随意设计
    closer_factor: float = 10.0  # 速度奖励系数
    collision_factor: float = 1000.0  # 重叠惩罚系数, >= 1 * reach finally
    reach_factor: float = 30  # 到达目标的奖励系数
    N_Agents: int = 10  # 训练时的agent数量
    N_Humans: int = 8
    Human_Height: float = 1.37
    Human_Mean_Speed: float = 1.34
    Human_Std_Speed: float = 0.37
    Agent_Max_Speed: float = 10.0  # m/s, agent的最大速度
    Scare_Factor: float = 25.0  # 恐惧因子, * 1/r * (1 + cos(vv)) * max(0, cos(rv))
    # 大致来说需要  s / r_min * 2 = reach_factor, s = reach_factor * r_min / 2
    # 同时, s / cutoff * 1 ~ closer_factor
    Cutoff_Scare_Distance: float = 5.0  # m , 超过这个距离就不考虑恐惧因子
    Communication_Delta_Time: float = 0.1  # s, 只影响前端渲染, 模拟时作为单位1
    World_Min_Bound: Tuple[int, int, int] = (0, 0, 0)  # m
    World_Max_Bound: Tuple[int, int, int] = (30, 10, 30)  # m
    enable_buildings: bool = True  # 是否启用建筑物
    enable_human_fear: bool = True  # 是否启用人类恐惧因子
    eval_max_timestamp: int = 200  # 评估时的最大通信次数(如果到达目标会继续生成新目标, 并且可能有不可达目标, 所以需要截断), 单位delta_t
    building_data: str = "train"  # 读取 {building_data}_buildings.csv 中的建筑物数据
    Cube_Min_Half_Length: float = 0.25  # 安全空间的最小半边长(飞机自身体积) 0.25
    nearby_agents_num: int = 5  # 观察时考虑的附近的agent数量

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingParam":
        tp = TrainingParam()
        for k, v in d.items():
            if hasattr(tp, k):
                setattr(tp, k, v)
            else:
                warnings.warn(f"Unknown TrainingParam key: {k}")
        return tp

    def __str__(self):
        return (
            f"TrainingParam(stage_name={self.stage_name}, N_Agents={self.N_Agents}, "
            f"N_Humans={self.N_Humans}, Human_Height={self.Human_Height}, "
            f"Human_Mean_Speed={self.Human_Mean_Speed}, Human_Std_Speed={self.Human_Std_Speed}, "
            f"Scare_Factor={self.Scare_Factor}, Cutoff_Scare_Distance={self.Cutoff_Scare_Distance}, "
            f"Communication_Delta_Time={self.Communication_Delta_Time}, "
            f"World_Min_Bound_Cube={self.World_Min_Bound}, "
            f"World_Max_Bound_Cube={self.World_Max_Bound}, "
            f"enable_buildings={self.enable_buildings}, enable_human_fear={self.enable_human_fear} "
            f"eval_max_timestamp={self.eval_max_timestamp}, building_data={self.building_data})"
            f"Cube_Min_Half_Length={self.Cube_Min_Half_Length}"
            f"nearby_agents_num={self.nearby_agents_num}"
        )

    def __post_init__(self):
        # 确保世界边界是元组
        self.World_Min_Bound = np.array(self.World_Min_Bound, dtype=np.float32)
        self.World_Max_Bound = np.array(self.World_Max_Bound, dtype=np.float32)
        if len(self.World_Min_Bound) != 3 or len(self.World_Max_Bound) != 3:
            raise ValueError("World bounds must be 3-dimensional tuples.")
