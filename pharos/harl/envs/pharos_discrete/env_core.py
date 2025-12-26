from collections import defaultdict
import os
from pathlib import Path
import traceback
from typing import List, Tuple, Dict

import numpy as np
from harl.envs.pharos_discrete.config import (
    SafeSpaceParamsContinuous,
    TrainingParam,
    action_to_move,
    MoveActionNum,
    SafeSpaceParams,
)
from harl.envs.pharos_discrete.agents.control import Agent
from harl.envs.pharos_discrete.agents.naive_control import NaiveControlAgent
from harl.envs.pharos_discrete.humans.base import Human
from harl.envs.pharos_discrete.humans.random_move_human import RandomMoveHuman
from harl.envs.pharos_discrete.render import AgentVis, HumanVis
from harl.envs.pharos_discrete.utils import (
    cuboid_intersection,
    normalize,
)
from harl.envs.pharos_discrete.buildings.building import (
    load_buildings,
    dis_to_cube,
    potential_to_cube,
    Building,
)


def generate_start_end(
    space_box: np.ndarray, buildings: list[Building], N: int, min_distance: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    生成N组起始点和目标点
    每个点需要满足以下条件：
    1. 不在建筑物内
    2. 在空间范围内
    3. 现有起始点不重合(目标点可以重合)

    :param space_box: 空间范围 (x_min, y_min, z_min, x_max, y_max, z_max)
    :param buildings: 建筑物列表
    :param N: 需要生成的起始点和目标点对数
    :param min_distance: 点和建筑物之间的最小距离
    """
    start_end_pairs = []
    starts = set()  # 用于存储已生成的起始点，避免重复
    dead_loop = 10000
    for _ in range(N):
        tries = 0
        while True:
            tries += 1
            if tries > dead_loop:
                raise RuntimeError(
                    f"Failed to generate start-end pairs after {dead_loop} attempts."
                )
            # 生成随机起始点
            start = np.random.randint(low=space_box[:3], high=space_box[3:], dtype=int)
            # 检查起始点是否在建筑物内
            if (
                any(
                    dis_to_cube(start, building.to_array()) < min_distance
                    for building in buildings
                )
                or tuple(start) in starts
            ):
                continue

            starts.add(tuple(start))  # 将起始点添加到集合中
            # 生成随机目标点
            end = np.random.randint(low=space_box[:3], high=space_box[3:], dtype=int)
            # 检查目标点是否在建筑物内
            if any(
                dis_to_cube(end, building.to_array()) < min_distance
                for building in buildings
            ):
                continue

            # 检查起始点和目标点是否重合
            if np.array_equal(start, end):
                continue

            start_end_pairs.append((start, end))
            break

    return start_end_pairs


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self, args: dict):
        # 尝试将args解析到TrainingParams
        self.tp: TrainingParam = TrainingParam.from_dict(args)
        self.agent_num = self.tp.N_Agents  # 设置智能体(小飞机)的个数
        self.human_num = self.tp.N_Humans  # 设置人类的个数
        self.nearby_num = self.tp.N_Agents - 1  # 设置智能体的观测范围
        self.tracing = False
        self.timestamp = 0
        self.action_dim = 1 if self.tp.action_discrete else 6
        self.space_box = np.array(
            [
                self.tp.World_Min_Bound_Cube[0] + 1,
                self.tp.World_Min_Bound_Cube[1] + 1,
                self.tp.World_Min_Bound_Cube[2] + 1,
                self.tp.World_Max_Bound_Cube[0] - 1,
                self.tp.World_Max_Bound_Cube[1] - 1,
                self.tp.World_Max_Bound_Cube[2] - 1,
            ]
        )
        if self.tp.enable_buildings:
            self.building_data = self.tp.building_data
            building_data_path = (
                Path("harl")
                / "envs"
                / "pharos_discrete"
                / "buildings"
                / f"{self.building_data}_buildings.csv"
            )
            if not building_data_path.exists():
                raise FileNotFoundError(
                    f"Building data file {building_data_path} does not exist."
                )
            self.buildings: List[Building] = load_buildings(str(building_data_path))
        else:
            self.buildings = []
        self.agent_vis_json_list: List[AgentVis] = []
        self.human_vis_json_list: List[AgentVis] = []
        self.env_done_timestamp = self.tp.eval_max_timestamp
        self.reset_agents()
        self.reset_humans()
        self.state_dim = self.get_state().shape[0]
        self.obs_dim = self.get_obs(self.agents[0]).shape[0]
        self.discrete = self.tp.action_discrete
        self.actions_dict = defaultdict(list)  # key: (ts, agent_id), value: action

    def _gen_randon_human_pos(self) -> np.ndarray:
        # HINT: 暂时不考虑建筑物对人的影响, 即人可以在建筑物里面
        pos_x = (
            np.random.random()
            * (self.tp.World_Max_Bound_Cube[0] - self.tp.World_Min_Bound_Cube[0])
            + self.tp.World_Min_Bound_Cube[0]
        )
        pos_y = self.tp.Human_Height / self.tp.Cube_Len
        pos_z = (
            np.random.random()
            * (self.tp.World_Max_Bound_Cube[2] - self.tp.World_Min_Bound_Cube[2])
            + self.tp.World_Min_Bound_Cube[2]
        )
        return np.array([pos_x, pos_y, pos_z])

    def reset_agents(self):
        # 生成智能体
        self.agents: List[Agent] = []
        pairs = generate_start_end(self.space_box, self.buildings, self.agent_num, 1)
        for i in range(self.agent_num):
            # 随机生成位置, 速度, 目标点
            init_velocity = np.zeros(3, dtype=np.int32)
            agent = NaiveControlAgent(
                id=i,
                name=f"agent_{i}",
                init_pos=pairs[i][0],
                init_velocity=init_velocity,
                target_pos=pairs[i][1],
                max_speed_cube=1,
            )
            self.agents.append(agent)

    def reset_humans(self):
        # HINT: 此处没有考虑human与建筑物的关系, 人是可以走入走出建筑物的, 暂时不影响
        self.humans: List[Human] = []
        for i in range(self.human_num):
            # 随机生成位置, 速度, 目标点
            pos = self._gen_randon_human_pos()
            human = RandomMoveHuman(
                id=i,
                position=pos,
                scare_factor=self.tp.Scare_Factor,
                mean_speed=self.tp.Human_Mean_Speed,
                std_speed=self.tp.Human_Std_Speed,
            )
            self.humans.append(human)

    def get_available_actions(self, training=True) -> List[np.ndarray] | None:
        # 训练时, 允许碰撞, 不允许穿过地面
        # avaliable_actions: bool(01) nparray (n_agents, action_dim)
        if not self.tp.action_discrete:
            # 连续动作空间, 返回None
            return None
        available_actions = np.ones((self.agent_num, MoveActionNum), dtype=np.float32)
        for idx, agent in enumerate(self.agents):
            for action in range(MoveActionNum):
                # 计算下一个位置
                move = action_to_move(action)
                next_pos = agent.pos + move
                # 检查是否穿过地面: y >= 0
                if next_pos[1] < 0:
                    available_actions[idx][action] = 0.0
        return [available_actions[i] for i in range(self.agent_num)]

    def get_state(self) -> np.ndarray:
        # global state
        positions = [a.pos for a in self.agents]  # 3n
        velocities = [a.velocity for a in self.agents]  # 3n
        target_positions = [a.target_pos for a in self.agents]  # 3n
        observations = [self.get_obs(a).tolist() for a in self.agents]  # obs_dim * n
        # 融合全局信息和局部信息
        max_bound = np.array(self.tp.World_Max_Bound_Cube)
        min_bound = np.array(self.tp.World_Min_Bound_Cube)
        for pos, vel, target_pos, obs in zip(
            positions, velocities, target_positions, observations
        ):
            pos = (pos - min_bound) / (max_bound - min_bound)
            vel = vel / (max_bound - min_bound)
            target_pos = (target_pos - min_bound) / (max_bound - min_bound)
            obs.extend(pos.tolist())
            obs.extend(vel.tolist())
            obs.extend(target_pos.tolist())
        state = []
        state.extend(max_bound.tolist())  # 3
        state.extend(min_bound.tolist())  # 3
        observations = np.concatenate(observations).flatten().tolist()
        state.extend(observations)
        return np.array(state)
        # return np.concatenate([self.get_obs(a) for a in self.agents]).flatten()

    def get_obs(self, agent: Agent) -> np.ndarray:
        K = self.nearby_num
        min_bound = np.array(self.tp.World_Min_Bound_Cube)
        max_bound = np.array(self.tp.World_Max_Bound_Cube)

        # 自身状态
        obs = []
        # id(加入异构策略)
        obs.append(agent.id)
        obs.extend(agent.pos / (max_bound - min_bound))
        obs.append(np.sum(np.abs(agent.pos - agent.target_pos)))
        # pos_norm = (agent.pos - min_bound) / (max_bound - min_bound)
        rel_goal_direction = normalize(
            (agent.target_pos - agent.pos) / (max_bound - min_bound)
        )

        # obs.extend(pos_norm.tolist())
        obs.extend(rel_goal_direction.tolist())

        # 查询最近K个邻居（不含自身）
        dists = [
            (np.linalg.norm(agent.pos - a.pos), a) for a in self.agents if a != agent
        ]
        dists.sort(key=lambda x: x[0])
        nearest = [a for _, a in dists[:K]]

        # 初始化邻居特征

        # 填充邻居特征
        # 这里有一个对称的问题, 基础的对称观察让双方互相等待对面
        directions = [
            np.array([1, 0, 0]),  # +x (right)
            np.array([-1, 0, 0]),  # -x (left)
            np.array([0, 1, 0]),  # +y (up)
            np.array([0, -1, 0]),  # -y (down)
            np.array([0, 0, 1]),  # +z (forward)
            np.array([0, 0, -1]),  # -z (backward)
        ]
        if self.tp.action_discrete:
            neighbor_features = np.zeros(6, dtype=np.float32)

            for i, direction in enumerate(directions):
                for a in nearest:
                    neighbor_pos = agent.pos + direction
                    if np.array_equal(a.pos, neighbor_pos):
                        # Case 1: Neighbor occupies one of the six directions
                        neighbor_features[i] += 1.0
                    elif np.linalg.norm(a.pos - neighbor_pos) == 1:
                        # Case 2: Neighbor is at the center of an edge
                        rel_velocity = np.dot(a.velocity - agent.velocity, direction)
                        if rel_velocity > 0:
                            neighbor_features[i] += 0.8  # Moving closer
                        elif rel_velocity == 0:
                            neighbor_features[i] += 0.5  # Stationary
                        else:
                            neighbor_features[i] += 0.2  # Moving away
            obs.extend(neighbor_features.tolist())
        else:
            # 连续动作空间使用最近的topk个邻居的位置和速度
            neighbor_features = []
            topk = 3
            nearest = nearest[:topk]
            for a in nearest:
                neighbor_features.append(a.pos / (max_bound - min_bound))
                neighbor_features.append(a.velocity / (max_bound - min_bound))
            neighbor_features = np.concatenate(neighbor_features).flatten().tolist()
            obs.extend(neighbor_features)

        # 计算给定安全空间后, agent如果按照安全空间飞, 可能的恐惧指数
        next_step_fears = []
        if self.tp.enable_human_fear:
            next_step_fears.append(self.cal_human_fear(agent.pos, agent.velocity))
        else:
            next_step_fears.append(0.0)
        for i, direction in enumerate(directions):
            possible_velocity = direction
            if self.tp.enable_human_fear:
                next_step_fear = (
                    self.cal_human_fear(
                        agent.pos + possible_velocity,
                        possible_velocity,
                    )
                    / self.tp.Scare_Factor
                )  # 缩放尺度到1
            else:
                next_step_fear = 0.0
            next_step_fears.append(next_step_fear)

        # 计算人类的恐惧惩罚, 重要的是相对值
        # min max 归一化
        next_step_fears = np.array(next_step_fears)
        # factor * 1/dij * (1 - cos(vi, vj))) * max(0, cos(vji, rji))
        step_fear_max = self.tp.Scare_Factor * 1 * 2 * 1
        next_step_fears = next_step_fears / step_fear_max  # 归一化
        obs.extend(next_step_fears)

        # 其他agent的斥力, 1/r^2
        agent_force_feat = np.zeros((3,), dtype=np.float32)
        for a in nearest:
            dist = np.linalg.norm(agent.pos - a.pos)
            if dist < 1e-6:
                continue
            agent_force_feat += (agent.pos - a.pos) / dist**3  # 也是1尺度
        # 归一化
        agent_force_feat = normalize(agent_force_feat)
        # 建筑的排斥力, 一步下的梯度(排斥力)
        # 不直接算力的原因是不容易算
        building_potentials = np.zeros((len(directions),), dtype=np.float32)
        assert building_potentials.shape[0] == 6, (
            f"building_potentials should have 6 elements, but {building_potentials.shape[0]} found."
        )
        for i, d in enumerate(directions):
            building_potentials[i] = sum(
                [potential_to_cube(agent.pos + d, b.to_array()) for b in self.buildings]
            )
        base_potential = sum(
            [potential_to_cube(agent.pos, b.to_array()) for b in self.buildings]
        )
        building_potentials -= (
            np.ones((building_potentials.shape), dtype=np.float32) * base_potential
        )
        # 势的梯度得到力
        # 再除以K到1量级
        K = 1
        building_potentials /= K

        obs.extend(agent_force_feat.tolist())
        obs.extend(building_potentials.tolist())
        return np.array(obs, dtype=np.float32)

    def reset(self) -> Tuple[list, list, list]:
        # obs, state, available_actions
        self.reset_agents()
        self.reset_humans()
        self.timestamp = 0
        return (
            [self.get_obs(a) for a in self.agents],
            [self.get_state() for _ in range(self.agent_num)],
            self.get_available_actions(),
        )
        # ATTENTION: 不要在此处调用reset_trace, logger中手动调用
        # 此处调用的话会在logger调用之前进行reset, 导致logger无法记录trace信息

    def enable_trace(self) -> None:
        # print("core: enable trace")
        self.tracing = True

    def disable_trace(self) -> None:
        # print("core: disable trace")
        self.tracing = False

    def reset_trace(self) -> None:
        # print("core: reset trace")
        print(f"tracing len: {len(self.agent_vis_json_list)} -> 0")
        self.agent_vis_json_list = []

        self.human_vis_json_list = []
        self.actions_dict = defaultdict(list)  # key: (ts, agent_id), value: action

    def cal_human_fear(
        self, agent_pos: np.ndarray, agent_velocity: np.ndarray
    ) -> float:
        """
        agent_pos: (3, )
        agent_velocity: (3, )
        """
        # (人类i对飞机j的)恐惧 = factor * 1/dij * (1 - cos(vi, vj))) * max(0, cos(vji, rji))
        # cos(vi, vj): 考虑人类朝向
        # cos(vj, rji): 考虑人类与飞机的相对位置, 拉远的行为应该不造成恐惧
        if np.linalg.norm(agent_velocity) < 1e-6:
            return 0.0
        fearness = 0
        # 矩阵化计算所有人类与该agent的恐惧值
        human_positions = np.array([h.position for h in self.humans])
        human_velocities = np.array([h.velocity for h in self.humans])
        scare_factors = np.array([h.scare_factor for h in self.humans])

        distances = np.linalg.norm(human_positions - agent_pos, axis=1)
        distances = np.maximum(distances, 0.1)  # 避免除0

        # 计算cosine相似度
        agent_velocity_norm = np.linalg.norm(agent_velocity)
        human_velocity_norms = np.linalg.norm(human_velocities, axis=1)

        # 避免除0
        agent_velocity_norm = max(agent_velocity_norm, 1e-6)
        human_velocity_norms = np.maximum(human_velocity_norms, 1e-6)
        dot_products = np.dot(human_velocities, agent_velocity)

        cosines_v = dot_products / (human_velocity_norms * agent_velocity_norm)

        rel_pos = agent_pos[np.newaxis, :] - human_positions  # 广播, rji
        rel_vel = agent_velocity[np.newaxis, :] - human_velocities  # vji
        rel_pos_norm = np.linalg.norm(rel_pos, axis=1)
        rel_vel_norm = np.linalg.norm(rel_vel, axis=1)

        cosines_r = (rel_vel * rel_pos).sum(axis=1) / (rel_vel_norm * rel_pos_norm)
        pos_angle_factor = np.maximum(0, cosines_r)  # max(0, cos(vji, rji))

        fearness = 1 / distances * scare_factors * (1 - cosines_v) * pos_angle_factor
        # 对于distance > cutoff_distance的恐惧值为0
        cutoff_distance = self.tp.Cutoff_Scare_Distance

        fearness = np.where(distances > cutoff_distance, 0, fearness)
        fearness_sum = np.sum(fearness)
        return fearness_sum

    def cal_reward(
        self,
        origin_pos: List[np.ndarray],
        origin_target_pos: List[np.ndarray],
        origin_safe_space_params: List[SafeSpaceParams]
        | List[SafeSpaceParamsContinuous],
        delta_reach_cnt: int,
    ) -> Tuple[float, dict]:
        # 1. 接近目标奖励
        # 2. 到达奖励
        # 3. 碰撞惩罚
        # 4. 人恐惧因子
        target_closer_reward = 0
        collision_penalty = 0

        arrive_reward = delta_reach_cnt * self.tp.reach_factor
        origin_pos = np.array(origin_pos)
        target_pos = np.array(origin_target_pos)
        origin_dis_sum = np.sum(np.linalg.norm(origin_pos - target_pos, axis=1))

        cur_pos = np.array([a.pos for a in self.agents])
        cur_dis_sum = np.sum(np.linalg.norm(cur_pos - target_pos, axis=1))
        target_closer_reward = (origin_dis_sum - cur_dis_sum) * self.tp.closer_factor

        a2a_collisions = 0
        a2h_collisions = 0
        a2b_collisions = 0
        if self.tp.action_discrete:
            assert isinstance(origin_safe_space_params[0], SafeSpaceParams), (
                "Expected SafeSpaceParams for discrete action space"
            )
            # 因为是离散的, 这个safe_space还是一个相对origin_pos的可移动向量
            # 计算所有的abs_safe_space的相交部分
            abs_safe_spaces_points: dict[np.ndarray, int] = {}
            # 计算两个安全空间的交集, 直接按照点集计算
            for i in range(self.agent_num):
                # 计算当前智能体的安全空间
                for point in origin_safe_space_params[i].to_points():
                    # 计算当前智能体的安全空间
                    point = tuple(point + self.agents[i].pos)
                    if point not in abs_safe_spaces_points:
                        abs_safe_spaces_points[point] = 1
                    else:
                        abs_safe_spaces_points[point] += 1

            collision_cube_cnt = 0
            for _, cnt in abs_safe_spaces_points.items():
                collision_cube_cnt += cnt - 1
                a2a_collisions += cnt - 1
            collision_penalty = collision_cube_cnt * self.tp.collision_factor
        else:
            # 连续动作空间, 计算所有的abs_safe_space的相交部分
            assert isinstance(origin_safe_space_params[0], SafeSpaceParamsContinuous), (
                "Expected SafeSpaceParamsContinuous for continuous action space"
            )
            abs_safe_spaces = [
                space.to_space() + self.agents[i].pos
                for i, space in enumerate(origin_safe_space_params)
            ]
            abs_safe_spaces = np.array(abs_safe_spaces)
            # 计算所有的abs_safe_space的相交部分
            collision_cube_cnt = 0
            for i in range(self.agent_num):
                for j in range(i + 1, self.agent_num):
                    # 计算两个abs_safe_space的相交部分
                    intersection = cuboid_intersection(
                        abs_safe_spaces[i], abs_safe_spaces[j]
                    )
                    a2a_collisions += intersection
                    collision_cube_cnt += intersection
            collision_penalty = collision_cube_cnt * self.tp.collision_factor

        # 以上是计算飞机之间的碰撞，下面计算飞机和其他物体的碰撞
        # 1. 人类
        for i in range(self.human_num):
            # 计算人类和飞机的距离
            human_pos = self.humans[i].position
            for j in range(self.agent_num):
                agent_pos = self.agents[j].pos
                # 计算人类和飞机的距离
                distance = np.linalg.norm(human_pos - agent_pos)
                if distance < 0.5:
                    collision_penalty += self.tp.collision_factor
                    a2h_collisions += 1

        # 2. 建筑物
        for building in self.buildings:
            # 计算建筑物和飞机的距离
            for j in range(self.agent_num):
                agent_pos = self.agents[j].pos
                distance = dis_to_cube(agent_pos, building.to_array())
                if distance < 0.5:
                    collision_penalty += self.tp.collision_factor
                    a2b_collisions += 1

        # 计算人类的恐惧惩罚
        fear_penalty = 0
        for i in range(self.agent_num):
            # 计算人类的恐惧惩罚
            if self.tp.enable_human_fear:
                fear_penalty += self.cal_human_fear(
                    self.agents[i].pos, self.agents[i].velocity
                )

        reward_info = {
            "target_closer_reward": target_closer_reward,
            "arrive_reward": arrive_reward,
            "collision_penalty": collision_penalty,
            "fear_penalty": fear_penalty,
            "a2a_collisions": a2a_collisions,
            "a2h_collisions": a2h_collisions,
            "a2b_collisions": a2b_collisions,
        }
        reward = target_closer_reward + arrive_reward - collision_penalty - fear_penalty
        return reward, reward_info

    def step(self, actions: List[np.ndarray]) -> list:
        """
        return local_obs, rewards, dones, infos
        """
        self.timestamp += 1
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            self.actions_dict[(self.timestamp, i)] = actions[i].tolist()
        # actions: save spaces
        if self.tp.action_discrete:
            spaces = [SafeSpaceParams.from_numpy(action) for action in actions]
        else:
            # HINT: 取max_move=1和离散比较
            spaces = [
                SafeSpaceParamsContinuous.from_numpy(action, max_move=1)
                for action in actions
            ]
        if self.tracing:
            # print("tracing in step")
            self.agent_vis_json_list.extend(self.agent_render_json(spaces))
            self.human_vis_json_list.extend(self.human_render_json())
        # 更新agent的位置
        delta_reach_cnt = 0
        origin_pos = [a.pos for a in self.agents]
        origin_target_pos = [a.target_pos for a in self.agents]
        for i in range(self.agent_num):
            new_pos, new_velocity = self.agents[i].estimate(spaces[i])
            if new_pos[1] <= 0:
                # 不允许穿过地面
                new_pos[1] = 0
                new_velocity[1] = 0
            self.agents[i].update(real_pos=new_pos, real_velocity=new_velocity)
            if self.agents[i].arrived:
                # 已经到达, 重新设置目标点
                # 会让环境永不done,
                delta_reach_cnt += 1  # HINT: 加这个奖励某种程度上在融入排队论
                self.agents[i].target_pos = generate_start_end(
                    self.space_box, self.buildings, 1, 1
                )[0][1]

                self.agents[i].velocity = np.zeros(3, dtype=np.int32)
                self.agents[i].arrived = False
        # 更新人类的位置
        for i in range(self.human_num):
            self.humans[i].update(delta_t=1)
            # 如果走出边界, 折返
            if any(
                self.humans[i].position > np.array(self.tp.World_Max_Bound_Cube)
            ) or any(self.humans[i].position < np.array(self.tp.World_Min_Bound_Cube)):
                self.humans[i].position = np.clip(
                    self.humans[i].position,
                    np.array(self.tp.World_Min_Bound_Cube),
                    np.array(self.tp.World_Max_Bound_Cube),
                )
                self.humans[i].velocity = -self.humans[i].velocity

        for i in range(self.agent_num):
            # 重新计算obs和done
            sub_agent_obs.append(self.get_obs(self.agents[i]))
            sub_agent_done.append(
                np.linalg.norm(self.agents[i].pos - self.agents[i].target_pos) < 1e-2
            )

        reward, reward_info = self.cal_reward(
            origin_pos=origin_pos,
            origin_target_pos=origin_target_pos,
            origin_safe_space_params=spaces,
            delta_reach_cnt=delta_reach_cnt,
        )
        # 共享reward
        sub_agent_reward = [[reward] for _ in range(self.agent_num)]
        sub_agent_info = [{**reward_info}] * self.agent_num
        if self.timestamp > self.env_done_timestamp:
            # HINT: 最大时间强制停止, 在eval时有用
            sub_agent_done = [True] * self.agent_num
        # print(f"step: {self.timestamp}, tracing: {self.tracing}")
        available_actions = self.get_available_actions()

        return [
            sub_agent_obs,
            sub_agent_reward,
            sub_agent_done,
            sub_agent_info,
            available_actions,
        ]

    def human_render_json(self) -> list[HumanVis]:
        return [
            HumanVis(
                hid=f"human/{h.id}",
                position=np.array([h.position[0], 0, h.position[2]])
                * self.tp.Cube_Len,  # 渲染的时候, 人的起始点是地面
                velocity=h.velocity
                * self.tp.Cube_Len
                / (self.tp.Communication_Delta_Time / 1000),  # m/s
                ts=int(self.timestamp * self.tp.Communication_Delta_Time),
            )
            for h in self.humans
        ]

    def agent_render_json(
        self, agent_spaces: List[SafeSpaceParams] | List[SafeSpaceParamsContinuous]
    ) -> list[AgentVis]:
        uids = [f"vehicle/{a.name}" for a in self.agents]
        positions = [a.pos for a in self.agents]
        velocities = [a.velocity for a in self.agents]
        ts = [self.timestamp] * self.agent_num

        include_areas = []  # xmin, ymin, zmin, xmax, ymax, zmax
        if self.tp.action_discrete:
            move_spaces = [s.to_space() for s in agent_spaces]
            new_positions = np.array(positions) + np.array(move_spaces)
            pos_arr = np.array(positions)
            # 计算包围盒
            for pos, new_pos in zip(pos_arr, new_positions):
                possible_points = []
                possible_points.append(np.array((0.5, 0.5, 0.5)) + pos)
                possible_points.append(np.array((0.5, 0.5, 0.5)) + new_pos)
                possible_points.append(np.array((-0.5, -0.5, -0.5)) + pos)
                possible_points.append(np.array((-0.5, -0.5, -0.5)) + new_pos)
                min_points = min(possible_points, key=lambda e: e[0] + e[1] + e[2])
                max_points = max(possible_points, key=lambda e: e[0] + e[1] + e[2])
                include_areas.append(
                    [
                        min_points[0],
                        min_points[1],
                        min_points[2],
                        max_points[0],
                        max_points[1],
                        max_points[2],
                    ]
                )
        else:
            # 连续动作空间, 直接使用safe space
            for i, a in enumerate(self.agents):
                include_areas.append(
                    (
                        agent_spaces[i].to_space().reshape(-1)
                        + np.array([a.pos, a.pos]).reshape(-1)
                    ).tolist()
                )
        target_positions = [a.target_pos for a in self.agents]

        include_areas = np.array(include_areas)
        # 将cube/timestamp单位转换回m/s
        vis_json_list = [
            AgentVis(
                u,
                p * self.tp.Cube_Len,  # m单位
                v * self.tp.Cube_Len / (self.tp.Communication_Delta_Time / 1000),
                int(t * self.tp.Communication_Delta_Time),  # ts ms单位
                i * self.tp.Cube_Len,  # m单位
                target_pos=tp * self.tp.Cube_Len,
                prev_action=self.actions_dict.get(
                    (t, idx), None
                ),  # 获取本agent上一个时刻的动作
            )
            for idx, (u, p, v, t, i, tp) in enumerate(
                zip(
                    uids,
                    positions,
                    velocities,
                    ts,
                    include_areas,
                    target_positions,
                )
            )
        ]
        return vis_json_list

    def seed(self, seed: int):
        np.random.seed(seed)

    def close(self):
        pass

    def get_tracing_info(self) -> Dict:
        # print("get tracing info")
        buildings = [b.to_vis_json() for b in self.buildings]
        for i, b in enumerate(buildings):
            bbox = b["bbox"]
            bbox = np.array(bbox) * self.tp.Cube_Len
            buildings[i]["bbox"] = bbox.tolist()
        return {
            "devices": self.agent_vis_json_list,
            "humans": self.human_vis_json_list,
            "buildings": buildings,
        }
