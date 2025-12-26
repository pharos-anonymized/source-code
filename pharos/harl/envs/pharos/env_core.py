from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from harl.envs.pharos.agents.control import Agent
from harl.envs.pharos.agents.naive_control import NaiveControlAgent
from harl.envs.pharos.humans.human import Human
from harl.envs.pharos.utils.utils import (
    SafeSpaceParams,
    calculate_overlap_volume,
    shortest_distance_between_trajectories,
    normalize,
)
from harl.envs.pharos.config import TrainingParam, RewardParams
from harl.envs.pharos.utils.render import AgentVis, HumanVis
from harl.envs.pharos.humans.randomwalk_human import RandomMoveHuman
from harl.envs.pharos.buildings.building import (
    load_buildings,
    dis_to_cube,
    Building,
    potential_to_cube,
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
            start = np.random.uniform(low=space_box[:3], high=space_box[3:]).astype(
                float
            )
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
            end = np.random.uniform(low=space_box[:3], high=space_box[3:]).astype(float)
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

    def __init__(self, args):
        self.tp = TrainingParam.from_dict(args)
        self.agent_num = self.tp.N_Agents  # 设置智能体(小飞机)的个数
        self.human_num = self.tp.N_Humans  # 设置人类的个数
        self.nearby_num = self.tp.nearby_agents_num  # 设置智能体的观测范围
        self.tp.World_Min_Bound = np.array(self.tp.World_Min_Bound, dtype=np.float32)
        self.tp.World_Max_Bound = np.array(self.tp.World_Max_Bound, dtype=np.float32)
        self.delta_t = self.tp.Communication_Delta_Time  # 设置智能体的通信时间间隔
        self.max_speed = self.tp.Agent_Max_Speed  # 设置智能体的最大速度
        self.space_box = np.array(
            [
                self.tp.World_Min_Bound[0],
                self.tp.World_Min_Bound[1],
                self.tp.World_Min_Bound[2],
                self.tp.World_Max_Bound[0],
                self.tp.World_Max_Bound[1],
                self.tp.World_Max_Bound[2],
            ],
            dtype=np.float32,
        )
        self.env_done_timestep = (
            self.tp.eval_max_timestamp * self.tp.Communication_Delta_Time * 1000
        )  # ms, 评估episode的持续时长(最大时间戳)
        self.agent_vis_json_list: List[AgentVis] = []
        self.human_vis_json_list: List[HumanVis] = []
        self.humans: List[Human] = []
        if self.tp.enable_buildings:
            self.building_data = self.tp.building_data
            building_data_path = (
                Path("harl")
                / "envs"
                / "pharos"
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
        print(f"Buildings: {self.buildings}")
        self.human_num = self.tp.N_Humans  # 设置人类的个数
        self.tracing = False
        self.reset()
        self.obs_dim = self.get_obs(self.agents[0]).shape[0]  # 每个agent的观测维度
        self.action_dim = 6
        self.state_dim = np.array(self.get_state()).shape[0]  # 全局状态维度
        self.action_dict = defaultdict(list)  # key: (ts, uid), value: action

    def get_state(self) -> np.ndarray:
        state = [self.get_obs(a) for a in self.agents]
        state = np.concatenate(state).flatten().tolist()
        state.extend(self.tp.World_Max_Bound)
        state.extend(self.tp.World_Min_Bound)
        return state

    def get_obs(self, agent: Agent) -> np.ndarray:
        top_k = self.nearby_num
        distances = [
            (np.linalg.norm(agent.pos - a.pos), idx)
            for idx, a in enumerate(self.agents)
        ]
        distances.sort()
        nearest_agents = [
            self.agents[idx] for _, idx in distances[: top_k + 1]
        ]  # +1: self
        # HINT: 应该学的是物理的v,pos等关系
        # obs 应该有 自己的pos(bound归一化), v, target_pos - pos, topk的运动物体轨迹间的最短距离和对应的时间
        # 时间用communication delta time 1/x加权, 距离用communication delta time * max_speed 1/x 计算
        # 如果距离小, 但时间长, 可以后面再考虑
        # 如果时间短(例如0), 但距离大, 则无所谓
        # 应该是一个类似乘积的结构, 减少一下特征维度, 这两个应该绑定的就不要解耦让网络学了
        obs = []
        obs.extend(
            (agent.pos - self.tp.World_Min_Bound)
            / (self.tp.World_Max_Bound - self.tp.World_Min_Bound)
        )
        obs.extend(agent.vertex)
        obs.append(np.sum(np.linalg.norm(agent.target_pos - agent.pos)))
        obs.extend(normalize(agent.target_pos - agent.pos))

        collide_factors = []

        for a in nearest_agents[1:]:
            dis, t = shortest_distance_between_trajectories(
                agent.pos, agent.vertex, a.pos, a.vertex
            )
            if dis < 1e-2:
                dis = 1e-2
            dis_factor = self.max_speed * self.delta_t / dis
            time_factor = self.delta_t / (self.delta_t + t)
            # 如果距离最小时, 时间在很远的地方, 导致 乘积更小的话, 那取现在的dis_factor和time_factor的乘积
            collide_factor = max(
                dis_factor * time_factor,
                self.max_speed * (self.delta_t / np.linalg.norm(a.pos - agent.pos)) * 1,
            )
            obs.append(collide_factor)
            collide_factors.append(collide_factor)

        # 计算给定安全空间后, agent如果按照安全空间飞, 可能的恐惧指数
        next_step_fears = []
        if self.tp.enable_human_fear:
            next_step_fears.append(self.cal_human_fear(agent.pos, agent.vertex))
        else:
            next_step_fears.append(0.0)
        directions = [
            np.array([1, 0, 0]),  # +x (right)
            np.array([-1, 0, 0]),  # -x (left)
            np.array([0, 1, 0]),  # +y (up)
            np.array([0, -1, 0]),  # -y (down)
            np.array([0, 0, 1]),  # +z (forward)
            np.array([0, 0, -1]),  # -z (backward)
        ]
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
        next_step_fears = np.array(next_step_fears)
        next_step_fears -= np.mean(next_step_fear)
        obs.extend(next_step_fears)

        # 其他agent的斥力, 1/r^2
        agent_force_feat = np.zeros((3,), dtype=np.float32)
        for a in nearest_agents[1:]:
            dist = np.linalg.norm(agent.pos - a.pos)
            if dist < 1e-6:
                continue
            agent_force_feat += (agent.pos - a.pos) / dist**3  # 也是1尺度
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
        for i in range(1, len(building_potentials)):
            building_potentials[i] -= base_potential  # 相对值
        obs.extend(agent_force_feat.tolist())
        obs.extend(building_potentials.tolist())

        obs.append(float(agent.max_speed))
        return np.array(obs, dtype=np.float32)

    def _gen_randon_human_pos(self) -> np.ndarray:
        # HINT: 暂时不考虑建筑物对人的影响, 即人可以在建筑物里面
        pos_x = (
            np.random.random()
            * (self.tp.World_Max_Bound[0] - self.tp.World_Min_Bound[0])
            + self.tp.World_Min_Bound[0]
        )
        pos_y = self.tp.Human_Height
        pos_z = (
            np.random.random()
            * (self.tp.World_Max_Bound[2] - self.tp.World_Min_Bound[2])
            + self.tp.World_Min_Bound[2]
        )
        return np.array([pos_x, pos_y, pos_z])

    def reset_agents(self):
        # 生成智能体
        self.agents: List[Agent] = []
        pairs = generate_start_end(self.space_box, self.buildings, self.agent_num, 1)
        for i in range(self.agent_num):
            # 随机生成位置, 目标点
            init_velocity = np.zeros(3, dtype=np.int32)
            agent = NaiveControlAgent(
                id=i,
                name="agent-%d" % i,
                pos=pairs[i][0],
                init_vertex=init_velocity,
                target_pos=pairs[i][1],
                max_speed=self.tp.Agent_Max_Speed,
                observation_agents=self.tp.nearby_agents_num,
                delta_t=self.tp.Communication_Delta_Time,
                Cube_Min_Half_Length=self.tp.Cube_Min_Half_Length,
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

    def reset(self):
        self.action_dict = defaultdict(list)
        self.timestamp = 0
        self.reset_agents()
        self.reset_humans()
        return [self.get_obs(agent) for agent in self.agents], [
            self.get_state()
        ] * self.agent_num

    def enable_trace(self) -> None:
        # print("core: enable trace")
        self.tracing = True

    def disable_trace(self) -> None:
        # print("core: disable trace")
        self.tracing = False

    def reset_trace(self) -> None:
        # print("core: reset trace")
        self.agent_vis_json_list = []
        self.action_dict = defaultdict(list)

    def cal_reward(
        self,
        last_abs_spaces: np.ndarray,
        origin_positions: np.ndarray,
        new_positions: np.ndarray,
        world_bounds: Tuple[np.ndarray, np.ndarray],
        param: RewardParams,
        delta_reach_cnt: int = 0,
    ) -> Tuple[float, Dict[str, Any]]:
        N = len(last_abs_spaces)
        # 所有agent的速度在target方向上的投影之和
        target_positions = np.array([agent.target_pos for agent in self.agents])
        distances_old = np.linalg.norm(origin_positions - target_positions, axis=1)
        distances_new = np.linalg.norm(new_positions - target_positions, axis=1)
        target_closer_reward = param.closer_factor * np.sum(
            distances_old - distances_new
        )
        # 2. 到达奖励
        reach_reward = delta_reach_cnt * param.reach_factor
        # 3. 计算重叠惩罚
        a2a_collisions = 0.0
        a2h_collisions = 0.0
        a2b_collisions = 0.0
        for i in range(N):
            for j in range(i + 1, N):  # 避免重复计算
                overlap_volume = calculate_overlap_volume(
                    last_abs_spaces[i], last_abs_spaces[j]
                )
                assert overlap_volume > -1e-2, "Overlap volume should be non-negative"
                # 根据重叠体积的大小给予不同程度的惩罚
                a2a_collisions += overlap_volume

        # 以上是计算飞机之间的碰撞，下面计算飞机和其他物体的碰撞
        # 1. 人类
        for i in range(self.human_num):
            # 计算人类和飞机安全空间的最小距离, >0
            human_pos = self.humans[i].position
            for j in range(self.agent_num):
                (x_min, y_min, z_min), (x_max, y_max, z_max) = last_abs_spaces[j]
                dis_to_human = dis_to_cube(
                    human_pos, np.array([x_min, y_min, z_min, x_max, y_max, z_max])
                )
                if dis_to_human <= 1e-2:
                    a2h_collisions += 1

        # 2. 建筑物
        for building in self.buildings:
            # 计算建筑物和飞机安全空间的重叠(如有)
            for j in range(self.agent_num):
                b_min, b_max = building.to_array()[:3], building.to_array()[3:]
                overlap_volume = calculate_overlap_volume(
                    last_abs_spaces[j], (b_min, b_max)
                )
                if overlap_volume > 0:
                    a2b_collisions += 0.05  # HINT: 这边的碰撞的惩罚系数调小, 因为建筑物太多了, 大惩罚无法训练

        # 计算人类的恐惧惩罚
        fear_penalty = 0
        for i in range(self.agent_num):
            # 计算人类的恐惧惩罚
            if self.tp.enable_human_fear:
                fear_penalty += self.cal_human_fear(
                    self.agents[i].pos, self.agents[i].vertex
                )
        overlap_penalty = (
            a2a_collisions * param.collision_factor
            + a2h_collisions * param.collision_factor
            + a2b_collisions * param.collision_factor
        )
        # 总奖励
        reward = target_closer_reward - overlap_penalty + reach_reward - fear_penalty

        info = {
            "target_closer_reward": target_closer_reward,
            "overlap_penalty": overlap_penalty,
            "reach_reward": reach_reward,
            "a2a_collisions": a2a_collisions,
            "a2h_collisions": a2h_collisions,
            "a2b_collisions": a2b_collisions,
            "fear_penalty": fear_penalty,
        }
        return reward, info

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

    def step(self, actions):
        """
        return local_obs, rewards, dones, infos
        """
        self.timestamp += self.tp.Communication_Delta_Time * 1000  # 1000/s
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            self.action_dict[(self.timestamp, i)] = actions[i].tolist()
        # actions: save spaces
        spaces = [SafeSpaceParams.from_numpy(action, self.tp) for action in actions]
        abs_spaces = [
            space.to_AABB() + (self.agents[idx].pos, self.agents[idx].pos)
            for idx, space in enumerate(spaces)
        ]
        delta_reach_cnt = 0
        origin_positions = [agent.pos for agent in self.agents]
        if self.tracing:
            # print("tracing in step")
            self.agent_vis_json_list.extend(self.agent_render_json(spaces))
            self.human_vis_json_list.extend(self.human_render_json())
        for i in range(self.agent_num):
            agent: NaiveControlAgent = self.agents[i]
            agent.time_consumed += self.timestamp * agent.delta_t / 1000
            # all other agents and obstacles
            new_pos, new_vertex = agent.estimate(spaces[i])
            if new_pos[1] <= 0:
                # 不允许穿过地面
                new_pos[1] = 0
                new_vertex[1] = 0
            agent.update(real_pos=new_pos, real_vertex=new_vertex)

            if not agent.moving:
                # 已经到达, 重新设置目标点
                # 会让环境永不done,
                delta_reach_cnt += 1  # HINT: 加这个奖励某种程度上在融入排队论
                agent.target_pos = generate_start_end(
                    self.space_box, self.buildings, 1, 1
                )[0][1]
                agent.init_pos = agent.pos
                agent.time_consumed = 0
                agent.vertex = np.zeros(
                    3,
                )
                agent.moving = True
                # 更新人类的位置
        for i in range(self.human_num):
            self.humans[i].update(delta_t=self.tp.Communication_Delta_Time)
            # 如果走出边界, 折返
            if any(self.humans[i].position > np.array(self.tp.World_Max_Bound)) or any(
                self.humans[i].position < np.array(self.tp.World_Min_Bound)
            ):
                self.humans[i].position = np.clip(
                    self.humans[i].position,
                    np.array(self.tp.World_Min_Bound),
                    np.array(self.tp.World_Max_Bound),
                )
                self.humans[i].velocity = -self.humans[i].velocity

        for i in range(self.agent_num):
            # 更新agent的位置后, 重新计算done和obs
            sub_agent_obs.append(self.get_obs(self.agents[i]))
            sub_agent_done.append(
                np.linalg.norm(self.agents[i].pos - self.agents[i].target_pos) < 1e-2
            )
        new_positions = [agent.pos for agent in self.agents]
        # 用last_abs_spaces来计算碰撞
        # 用更新后的速度等和到达目标数量来计算reward
        reward, reward_info = self.cal_reward(
            last_abs_spaces=np.array(abs_spaces),
            origin_positions=np.array(origin_positions),
            new_positions=np.array(new_positions),
            world_bounds=(self.tp.World_Min_Bound, self.tp.World_Max_Bound),
            param=RewardParams(
                closer_factor=self.tp.closer_factor,
                collision_factor=self.tp.collision_factor,
                reach_factor=self.tp.reach_factor,
            ),
            delta_reach_cnt=delta_reach_cnt,
        )
        # 共享reward
        sub_agent_reward = [[reward] for _ in range(self.agent_num)]
        sub_agent_info = [{**reward_info}] * self.agent_num
        if self.timestamp > self.env_done_timestep:
            # HINT: 最大时间强制停止, 在eval时有用
            sub_agent_done = [True] * self.agent_num
        if self.tracing:
            self.agent_vis_json_list.extend(self.agent_render_json(spaces))

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def agent_render_json(self, spaces: List[SafeSpaceParams]) -> list[AgentVis]:
        uids = [f"vehicle/{a.name}" for a in self.agents]
        positions = [a.pos for a in self.agents]
        velocities = [a.vertex for a in self.agents]
        ts = [self.timestamp] * self.agent_num
        abs_spaces = [
            np.array(space.to_AABB()) + (self.agents[idx].pos, self.agents[idx].pos)
            for idx, space in enumerate(spaces)
        ]
        include_areas = [np.concatenate(space) for space in abs_spaces]
        target_pos = [a.target_pos for a in self.agents]
        vis_json_list = [
            AgentVis(u, p, v, t, i, target_pos=tp, prev_action=self.action_dict[(t, u)])
            for u, p, v, t, i, tp in zip(
                uids, positions, velocities, ts, include_areas, target_pos
            )
        ]
        return vis_json_list

    def human_render_json(self) -> list[HumanVis]:
        return [
            HumanVis(
                hid=f"human/{h.id}",
                position=np.array([h.position[0], 0, h.position[2]]),  # y=0
                velocity=h.velocity,
                ts=int(self.timestamp),  # ms
            )
            for h in self.humans
        ]

    def seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def close(self):
        pass

    def get_tracing_info(self) -> Dict:
        # print("get tracing info")
        buildings = [b.to_vis_json() for b in self.buildings]
        for i, b in enumerate(buildings):
            bbox = b["bbox"]
            bbox = np.array(bbox)
            buildings[i]["bbox"] = bbox.tolist()
        return {
            "devices": self.agent_vis_json_list,
            "humans": self.human_vis_json_list,
            "buildings": buildings,
        }
