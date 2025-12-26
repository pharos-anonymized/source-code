import warnings
from dataclasses import dataclass, field
from typing import Tuple
from enum import Enum
import numpy as np
import torch


@dataclass
class RewardParams:
    closer_factor: float = 10.0  # 速度奖励系数
    collision_factor: float = 1000.0  # 重叠惩罚系数, >= 1 * reach finally
    reach_factor: float = 30  # 到达目标的奖励系数


@dataclass
class TrainingParam:
    # ATTENTION: 空间参数需要按照cube缩放以适应不同尺度
    # 也就是说, 所有的速度等初始单位都不是m/s, 而是以cube/s
    # 同时, 调度的最小单位是cube: 无法进行cube以下粒度的避障保证
    # (系统控制的粒度是cube, 而cube内部的飞行是由飞机自身的飞控算法决定的,
    # 鉴于非侵入式, 如果想要一个大cube如 10m, 则所有避障/控制对人的惊吓都需要以10m为粒度)
    # 推荐粒度为1m, 这是由10m/s的agent最大速度和100ms的通信时延得到的
    Cube_Len: float = 1.0
    # 每个小立方体的边长(m), 所有的位置都是以cube为单位, 速度以cube/delta_t为单位
    stage_name: str = "stage1"  # 训练阶段名称, 随意设计
    closer_factor: float = 10.0  # 速度奖励系数
    collision_factor: float = 1000.0  # 重叠惩罚系数, >= 1 * reach finally
    reach_factor: float = 30  # 到达目标的奖励系数
    N_Agents: int = 10  # 训练时的agent数量
    N_Humans: int = 8
    Human_Height: float = field(
        init=False
    )  # 不允许直接传入构造函数，用__post_init__赋值
    Human_Mean_Speed: float = field(init=False)
    Human_Std_Speed: float = field(init=False)
    Scare_Factor: float = 25.0  # 恐惧因子, * 1/r * (1 + cos(vv)) * max(0, cos(rv))
    # 大致来说需要  s / r_min * 2 = reach_factor, s = reach_factor * r_min / 2
    # 同时, s / cutoff * 1 ~ closer_factor
    Cutoff_Scare_Distance: float = 5.0  # cube , 超过这个距离就不考虑恐惧因子
    Communication_Delta_Time: int = 100  # 只影响前端渲染, 模拟时作为单位1
    # ms, 不影响训练, 只影响渲染时每次状态变化的前端时间间隔
    World_Min_Bound_Cube: Tuple[int, int, int] = (0, 0, 0)
    World_Max_Bound_Cube: Tuple[int, int, int] = (30, 10, 30)
    Agent_Max_Step_Cube: int = 2
    # 每次通信最多移动X-1个Min_Cube_Len, 目前只支持一次一个, 离散状态下启用
    enable_buildings: bool = True  # 是否启用建筑物
    enable_human_fear: bool = True  # 是否启用人类恐惧因子
    action_discrete: bool = True  # 是否使用离散动作空间
    eval_max_timestamp: int = 200  # 评估时的最大通信次数(如果到达目标会继续生成新目标, 并且可能有不可达目标, 所以需要截断), 单位delta_t
    building_data: str = "train"  # 读取 {building_data}_buildings.csv 中的建筑物数据

    def __post_init__(self):
        self.Human_Height = 1.7 / self.Cube_Len  # cube, 全部按照cube缩放
        self.Human_Mean_Speed = (1.34 / self.Cube_Len) * (
            self.Communication_Delta_Time / 1000
        )  # cube/delta_t, 平均速度
        self.Human_Std_Speed = (0.37 / self.Cube_Len) * (
            self.Communication_Delta_Time / 1000
        )  # 1.34，0.37是成年人步行速度的统计值

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
            f"TrainingParam in stage: {self.stage_name}, "
            f"Reward Params: "
            f"- closer_factor: {self.closer_factor}, "
            f"- collision_factor: {self.collision_factor}, "
            f"- reach_factor: {self.reach_factor}, "
            f"Environment Params: "
            f"- N_Agents: {self.N_Agents}, "
            f"- N_Humans: {self.N_Humans}, "
            f"  - Human_Height: {self.Human_Height}, "
            f"  - Scare_Factor: {self.Scare_Factor}, "
            f"  - Cutoff_Scare_Distance: {self.Cutoff_Scare_Distance}, "
            f"  - Human_Mean_Speed: {self.Human_Mean_Speed}, "
            f"  - Human_Std_Speed: {self.Human_Std_Speed}, "
            f"- Communication_Delta_Time: {self.Communication_Delta_Time}, "
            f"- Cube_Len: {self.Cube_Len}, "
            f"- World_Min_Bound_Cube: {self.World_Min_Bound_Cube}, "
            f"- World_Max_Bound_Cube: {self.World_Max_Bound_Cube}, "
            f"- Agent_Max_Step_Cube: {self.Agent_Max_Step_Cube}, "
            f"- enable_buildings: {self.enable_buildings}"
            f"- enable_human_fear: {self.enable_human_fear}"
            f"- eval_max_timestamp: {self.eval_max_timestamp}, "
            f"Action Params: "
            f"- action_discrete: {self.action_discrete}, "
            f"Data Params: "
            f"- building_data: {self.building_data}"
        )


_move = {
    "STAY": np.array([0, 0, 0]),
    "UP": np.array([0, 0, 1]),
    "DOWN": np.array([0, 0, -1]),
    "LEFT": np.array([-1, 0, 0]),
    "RIGHT": np.array([1, 0, 0]),
    "FORWARD": np.array([0, 1, 0]),
    "BACKWARD": np.array([0, -1, 0]),
}
MoveActionNum = len(_move)  # 动作数量


def action_to_move(action: int) -> np.ndarray:
    """
    将动作转换为移动向量
    :param action: 动作编号
    :return: 移动向量
    """
    if action == 0:
        return _move["STAY"]  # 默认停留不动
    elif action == 1:
        return _move["UP"]
    elif action == 2:
        return _move["DOWN"]
    elif action == 3:
        return _move["LEFT"]
    elif action == 4:
        return _move["RIGHT"]
    elif action == 5:
        return _move["FORWARD"]
    elif action == 6:
        return _move["BACKWARD"]
    warnings.warn(
        f"Invalid action: {action}. Must be between 0 and {MoveActionNum - 1}."
    )
    print(action)
    assert False


@dataclass
class SafeSpaceParams:
    move_action: int = 0
    distance: int = 1

    def to_space(self) -> np.ndarray:
        move_direction = action_to_move(self.move_action)
        return move_direction * self.distance

    def to_points(self) -> set[Tuple[int, int, int]]:
        """
        将SafeSpaceParams转换为点集
        :return: 点集
        """
        move_direction = action_to_move(self.move_action)
        points = set()
        for i in range(self.distance + 1):
            points.add(tuple((move_direction * i).astype(int)))
        return points

    @staticmethod
    def from_numpy(arr: np.ndarray[(1,), int]) -> "SafeSpaceParams":
        assert len(arr) == 1
        # TODO: 完成multidiscrete的适配, 无法用一维模拟二维
        direction = arr[0]
        distance = 0 if direction == 0 else 1
        return SafeSpaceParams(move_action=direction, distance=distance)


@dataclass
class SafeSpaceParamsContinuous:
    # all >= 0.5
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def to_space(self, moveable: bool = False) -> np.ndarray:
        # shape: (2, 3)
        # moveable: 如果为真, 返回的空间需要减去不可移动的0.5Cube边界
        if not moveable:
            return np.array(
                [
                    [-self.x_min, -self.y_min, -self.z_min],
                    [self.x_max, self.y_max, self.z_max],
                ]
            )
        else:
            return np.array(
                [
                    [-self.x_min + 0.5, -self.y_min + 0.5, -self.z_min + 0.5],
                    [self.x_max - 0.5, self.y_max - 0.5, self.z_max - 0.5],
                ]
            )

    @staticmethod
    def from_numpy(
        arr: np.ndarray[(6,), float], max_move: float = 1
    ) -> "SafeSpaceParams":
        arr = torch.sigmoid(torch.tensor(arr))
        # 再缩放到目标区间[0.5, max_move + 0.5]
        arr = arr * max_move + 0.5
        assert len(arr) == 6, "Input array must have 6 elements."
        return SafeSpaceParamsContinuous(
            x_min=arr[0],
            x_max=arr[1],
            y_min=arr[2],
            y_max=arr[3],
            z_min=arr[4],
            z_max=arr[5],
        )
