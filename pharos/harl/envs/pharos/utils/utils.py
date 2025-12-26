import logging
from typing import Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
from sympy.stats.sampling.sample_numpy import numpy
from tensorboard.plugins.scalar.summary import scalar
from torch.ao.nn.quantized.functional import threshold

from harl.envs.pharos.config import TrainingParam


def sigmoid(x, bias=0, k=1):
    """Sigmoid函数，能够将输入映射到(0, 1)的范围。"""
    x = np.clip(x, bias - 10, bias + 10)  # 防止溢出, exp(10)=2.3w已经很大了
    # k越小, 函数在bias附近越平缓
    return 1 / (1 + np.exp(-k * (x - bias)))


@dataclass
class SafeSpaceParams:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def to_AABB(self) -> np.ndarray:
        return np.array(
            [(self.x_min, self.y_min, self.z_min), (self.x_max, self.y_max, self.z_max)]
        )

    @staticmethod
    def from_numpy(arr: np.ndarray, tp: TrainingParam) -> "SafeSpaceParams":
        lens = np.maximum(arr[:3], 0)
        alphas = [sigmoid(x) for x in arr[3:]]
        reasonable_max = tp.Communication_Delta_Time * tp.Agent_Max_Speed
        return SafeSpaceParams(
            x_min=-max(lens[0] * (1 - alphas[0]), reasonable_max)
            - tp.Cube_Min_Half_Length,
            x_max=max(lens[0] * alphas[0], reasonable_max) + tp.Cube_Min_Half_Length,
            y_min=-max(lens[1] * (1 - alphas[1]), reasonable_max)
            - tp.Cube_Min_Half_Length,
            y_max=max(lens[1] * alphas[1], reasonable_max) + tp.Cube_Min_Half_Length,
            z_min=-max(lens[2] * (1 - alphas[2]), reasonable_max)
            - tp.Cube_Min_Half_Length,
            z_max=max(lens[2] * alphas[2], reasonable_max) + tp.Cube_Min_Half_Length,
        )


def cuboid_distance(cuboid1, cuboid2):
    """
    计算两个长方体之间最近的距离。

    Args:
        cuboid1: 第一个长方体的坐标，格式为 ((x_min, y_min, z_min), (x_max, y_max, z_max))。
        cuboid2: 第二个长方体的坐标，格式为 ((x_min, y_min, z_min), (x_max, y_max, z_max))。

    Returns:
        两个长方体之间最近的距离。
    """

    # 将输入转换为numpy数组以便于计算
    cuboid1 = np.array(cuboid1)
    cuboid2 = np.array(cuboid2)

    # 计算中心点坐标
    center1 = (cuboid1[0] + cuboid1[1]) / 2
    center2 = (cuboid2[0] + cuboid2[1]) / 2

    # 计算各维度上的距离
    distances = np.abs(center1 - center2)
    # 计算各维度上的重叠长度
    edge_lens1 = cuboid1[1] - cuboid1[0]
    edge_lens2 = cuboid2[1] - cuboid2[0]
    overlaps = (edge_lens1 + edge_lens2) / 2 - distances

    # 计算最近距离
    distance = 0
    for i in range(3):
        if overlaps[i] < 0:  # 没有重叠
            distance += overlaps[i] ** 2

    return np.sqrt(distance) if distance > 0 else 0


def calculate_overlap_volume(space1, space2):
    """
    计算两个 AABB 立方体的重叠体积
    """
    s1_min, s1_max = space1
    s2_min, s2_max = space2

    # 计算每个维度上的重叠长度
    dx = max(0, min(s1_max[0], s2_max[0]) - max(s1_min[0], s2_min[0]))
    dy = max(0, min(s1_max[1], s2_max[1]) - max(s1_min[1], s2_min[1]))
    dz = max(0, min(s1_max[2], s2_max[2]) - max(s1_min[2], s2_min[2]))

    # 重叠体积
    return dx * dy * dz


def calculate_volume(space):
    """
    计算 AABB 立方体的体积
    """
    s_min, s_max = space
    return np.prod(s_max - s_min)


def normalize(arr: np.ndarray):
    """
    Normalize the input array.
    Args:
        arr: The input array.
    Returns:
        The normalized array.
    """
    if np.linalg.norm(arr) < 1e-2:
        return normalize(
            np.random.randn(*arr.shape)
        )  # 如果向量长度过小, 则随机生成一个单位向量
    return arr / np.linalg.norm(arr)


def ray_aabb_intersection(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
):
    """
    计算射线与 AABB 的交点.
    射线: R(t) = ray_origin + t * ray_direction (t >= 0)
    安全空间: AABB Box: [aabb_min, aabb_max]
    why work:
    - 分别计算每个维度上的交点, 然后最大的t就是出点, 最小的t(会是负数)就是射线的另一端
    - 由于R(t)线性, 所有维度上的出点中, t最小的必然是实际交点.
    参数:
    aabb_min (np.ndarray): AABB 的最小点 (3D 坐标).
    aabb_max (np.ndarray): AABB 的最大点 (3D 坐标).
    ray_origin (np.ndarray): 射线的起点 (3D 坐标).
    ray_direction (np.ndarray): 射线的方向向量 (3D 坐标), 必须归一化.

    返回值:
    np.ndarray: 如果相交, 返回交点 (3D 坐标); 否则, 返回 None.
    """

    t_min = np.full(3, -np.inf)
    t_max = np.full(3, np.inf)
    epsilon = 1e-6
    for i in range(3):
        if abs(ray_direction[i]) < epsilon:
            # 如果 ray_direction[i] 接近零，则射线与该轴平行
            # 如果射线起点在该轴上的 AABB 范围之外，则不相交
            if ray_origin[i] < aabb_min[i] or ray_origin[i] > aabb_max[i]:
                return None
            # 否则，忽略该轴
            continue
        inv_dir = 1.0 / ray_direction[i]
        t0 = (aabb_min[i] - ray_origin[i]) * inv_dir
        t1 = (aabb_max[i] - ray_origin[i]) * inv_dir

        if inv_dir < 0:
            t0, t1 = t1, t0

        t_min[i] = t0
        t_max[i] = t1

    t_enter = np.max(t_min)
    t_exit = np.min(t_max)

    if t_exit < t_enter or t_exit < 0:
        return None

    t = t_enter
    if t < 0:
        t = t_exit

    intersection = ray_origin + t * ray_direction
    return intersection


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    # 计算点积
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    dot_product = np.dot(v1, v2)
    # 计算每个向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 计算夹角（以弧度表示）
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))

    return angle


def shortest_distance_between_trajectories(
    p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray
) -> Tuple[float, float]:
    """
    计算两个运动物体轨迹间的最短距离和对应的时间t
    p1, v1: 第一个物体的初始位置和速度向量
    p2, v2: 第二个物体的初始位置和速度向量
    返回: (最短距离, t)
    """

    # 计算相对位置和相对速度
    dp = p2 - p1
    dv = v2 - v1

    # 计算最优时间t
    # 通过求导可得：t = -dp·dv / (dv·dv)
    dv_dot_dv = np.dot(dv, dv)

    if dv_dot_dv == 0:
        # 如果相对速度为0，说明两物体保持相对静止
        distance = np.linalg.norm(dp)
        return distance, 0

    t = -np.dot(dp, dv) / dv_dot_dv

    # 如果计算出的t<=0，则最小距离出现在t=0时
    t = max(0, t)

    # 计算t时刻的距离
    distance = np.linalg.norm(dp + dv * t)

    return distance, t


import numpy as np


def process_rects(rects: np.ndarray, max_iterations=1000) -> np.ndarray:
    """
    处理长方体的相交关系，通过切分相交部分使得长方体组不再相交

    参数:
    rects: 形状为(N, 2, 3)的numpy数组，代表N个长方体的左下角和右上角坐标
    max_iterations: 最大迭代次数，防止无限循环

    返回:
    新的rects数组，代表处理后的长方体
    """
    N = rects.shape[0]
    result_rects = rects.copy()

    # 记录迭代次数
    iterations = 0

    # 记录是否有修改
    modified = True

    while modified and iterations < max_iterations:
        modified = False
        iterations += 1

        # 遍历所有长方体对
        for i in range(N):
            for j in range(i + 1, N):
                # 判断长方体i和j是否相交
                if is_intersect(result_rects[i], result_rects[j]):
                    # 判断特殊情况
                    if is_contained(result_rects[i], result_rects[j]):
                        # A完全在B内部，不切分
                        continue
                    if is_center_contained(result_rects[i], result_rects[j]):
                        # A的中心在B内部，不切分
                        continue
                    if is_contained(result_rects[j], result_rects[i]):
                        # B完全在A内部，不切分
                        continue
                    if is_center_contained(result_rects[j], result_rects[i]):
                        # B的中心在A内部，不切分
                        continue

                    # 切分长方体
                    new_rect_i, new_rect_j = cut_intersection(
                        result_rects[i], result_rects[j]
                    )

                    # 检查是否有修改
                    if not np.array_equal(
                        new_rect_i, result_rects[i]
                    ) or not np.array_equal(new_rect_j, result_rects[j]):
                        result_rects[i] = new_rect_i
                        result_rects[j] = new_rect_j
                        modified = True

                    # 如果有修改，跳出内层循环
                    if modified:
                        break

            # 如果有修改，跳出外层循环
            if modified:
                break

    return result_rects


def is_intersect(rect1, rect2):
    """判断两个长方体是否相交"""
    # 检查在每个维度上是否有重叠
    for dim in range(3):
        if rect1[1][dim] <= rect2[0][dim] or rect2[1][dim] <= rect1[0][dim]:
            return False
    return True


def is_contained(rect1, rect2):
    """判断长方体rect1是否完全在长方体rect2内部"""
    # 检查rect1是否在每个维度上都被rect2包含
    for dim in range(3):
        if rect1[0][dim] < rect2[0][dim] or rect1[1][dim] > rect2[1][dim]:
            return False
    return True


def is_center_contained(rect1, rect2):
    """判断长方体rect1的中心是否在长方体rect2内部"""
    # 计算rect1的中心
    center = (rect1[0] + rect1[1]) / 2

    # 检查center是否在rect2内部
    for dim in range(3):
        if center[dim] < rect2[0][dim] or center[dim] > rect2[1][dim]:
            return False
    return True


def cut_intersection(rect1, rect2):
    """切分两个相交长方体的相交部分"""
    # 计算相交部分
    intersection_min = np.maximum(rect1[0], rect2[0])
    intersection_max = np.minimum(rect1[1], rect2[1])

    # 计算相交部分的各维度长度
    delta = intersection_max - intersection_min

    # 找出最小的相交维度
    min_dim = np.argmin(delta)

    # 创建新的长方体
    new_rect1 = rect1.copy()
    new_rect2 = rect2.copy()

    # 计算两个长方体的体积
    volume1 = np.prod(rect1[1] - rect1[0])
    volume2 = np.prod(rect2[1] - rect2[0])

    # 选择切分体积较小的长方体
    if volume1 <= volume2:
        # 切分rect1
        if rect1[0][min_dim] < rect2[0][min_dim]:
            # rect1的右边部分与rect2相交，修改rect1的上界
            new_rect1[1][min_dim] = rect2[0][min_dim]
        else:
            # rect1的左边部分与rect2相交，修改rect1的下界
            new_rect1[0][min_dim] = rect2[1][min_dim]
    else:
        # 切分rect2
        if rect2[0][min_dim] < rect1[0][min_dim]:
            # rect2的右边部分与rect1相交，修改rect2的上界
            new_rect2[1][min_dim] = rect1[0][min_dim]
        else:
            # rect2的左边部分与rect1相交，修改rect2的下界
            new_rect2[0][min_dim] = rect1[1][min_dim]

    return new_rect1, new_rect2
