import argparse
import json
from pathlib import Path
import time
from typing import override
import numpy as np
from gym import Env
import torch
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.actors.mappo import MAPPO
from harl.models.base.plain_mlp import PlainMLP
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.envs.pharos_discrete.env_core import EnvCore
import pyomo.environ as pyo
from pyomo.environ import value
import math
from termcolor import colored

import torch.nn as nn

"""
solver: conda install -c conda-forge ipopt
"""


def _t2n(value):
    """Convert torch.Tensor to numpy.ndarray."""
    return value.detach().cpu().numpy()


def print_model_info(model: pyo.ConcreteModel):
    print("===== Variables =====")
    for v in model.component_data_objects(pyo.Var, descend_into=True):
        val = v.value
        if val is None:
            print(f"Variable {v.name} Index: {v.index()} has no initial value.")
        else:
            print(f"Variable {v.name} Index: {v.index()} Value: {val}")

    print("\n===== Constraints =====")
    for c in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        print(f"Constraint {c.name} Index: {c.index()} Expression: {c.expr}")
        print(f"  Lower bound: {c.lower}, Upper bound: {c.upper}")

    # 打印总的约束个数和变量个数
    num_constraints = sum(1 for _ in model.component_data_objects(pyo.Constraint))
    num_vars = sum(1 for _ in model.component_data_objects(pyo.Var))
    print(
        colored(
            f"\nTotal Constraints: {num_constraints}, Total Variables: {num_vars}",
            "yellow",
        )
    )


class GenericSolver:
    def solve_single_step(self, env: EnvCore) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")


class TorchModelInit:
    """
    A base class for initializing Torch models.
    This class can be extended to include specific model initialization logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_model(self):
        """
        Initialize the model parameters.
        This method should be overridden by subclasses to implement specific initialization logic.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TorchModelInitActor(TorchModelInit):
    def __init__(
        self,
        algo_args: dict,
        observation_space,
        action_space,
        model_dir,
        model_id,
        algo_name: str = "mappo",
        *args,
        **kwargs,
    ):
        super().__init__(algo_name, *args, **kwargs)
        self.algo_name = algo_name
        self.actor = ALGO_REGISTRY[algo_name](
            {**algo_args["model"], **algo_args["algo"]},
            observation_space,
            action_space,
            device=get_device(),
        )
        self.model_dir = model_dir
        self.model_id = model_id

    def init_model(self):
        """
        Initialize the actor model parameters.
        """
        print(f"Initializing actor model with {self.algo_name} algorithm.")
        return self.actor.restore(self.model_dir, self.model_id)


class PretrainedRLSolver(GenericSolver):
    """
    A solver that uses a pretrained RL model to solve a single step.
    """

    def __init__(
        self,
        model_dir: str,
        log_dir: str,
        vis_dir: str,
        actor_init: TorchModelInit,
        critic_init: TorchModelInit | None = None,
        value_normalizer_init: TorchModelInit | None = None,
    ):
        super().__init__(name="PretrainedRLSolver")
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.vis_dir = vis_dir
        self.actors, self.critic, self.value_normalizer = self.restore(
            model_dir=model_dir,
            actor_init=TorchModelInitActor,
        )

    @override
    def solve_single_step(self, env: EnvCore):
        """
        Solve a single step using the pretrained RL model.
        """
        # TODO
        pass

    def restore(
        model_dir: str,
        actor_init: TorchModelInit,
        critic_init: TorchModelInit,
        value_normalizer_init: TorchModelInit | None,
    ) -> None:
        """Restore model parameters."""
        print("Restoring model from {}".format(model_dir))
        agent_ids = []
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        agent_ids = [
            int(p.name.split("_")[-1].removeprefix("agent"))
            for p in Path(model_dir).glob("actor_agent*.pth")
        ]
        if not agent_ids:
            raise FileNotFoundError(f"No actor models found in {model_dir}.")
        print(f"Found agent IDs: {agent_ids}")
        actors = [actor_init(id=id).init_model() for id in agent_ids]
        critic = critic_init().init_model()
        value_normalizer = (
            value_normalizer_init().init_model() if value_normalizer_init else None
        )
        for id, a in zip(agent_ids, actors):
            policy_actor_state_dict = torch.load(
                str(model_dir) + "/actor_agent" + str(id) + ".pt"
            )
            a.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(model_dir) + "/critic_agent" + ".pt"
            )
            critic.load_state_dict(policy_critic_state_dict)
            if value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(model_dir) + "/value_normalizer" + ".pt"
                )
                value_normalizer.load_state_dict(value_normalizer_state_dict)
        return actors, critic, value_normalizer

    @torch.no_grad()
    def eval(
        self, env: EnvCore, episodes: int = 1, timesteps: int = 50, seed: int = 42
    ):
        episode = 0
        env.seed(seed)
        for episode in range(episodes):
            obs, states, avaliable_actions = env.reset()
            done = False
            step = 0
            start_time = time.time()
            while not done and step < timesteps:
                step += 1

                actions = []
                for actor in self.actors:
                    action = actor(obs)
                    actions.append(_t2n(action))
                obs, rewards, done, infos, avaliable_actions = env.step(actions)
                # Here you can log metrics or visualize the environment
                print(f"Episode {episode}, Rewards: {rewards}")
                tracing = env.get_tracing_info()
            print("Episode {} finished after {} steps.".format(episode, step))
            print(
                "Time taken for episode {}: {:.2f} seconds".format(
                    episode, time.time() - start_time
                )
            )


class PyomoSolver(GenericSolver):
    """
    A solver that uses mathematical methods to solve a single step.
    This is a placeholder for actual mathematical solving logic.
    """

    def __init__(self, name: str = "ipopt"):
        self.name = name
        self.solver = pyo.SolverFactory(name)
        if name == "ipopt":
            self.solver.options["tol"] = 1e-6
            self.solver.options["max_iter"] = 1000
            self.solver.options["linear_solver"] = "mumps"
            self.solver.options["halt_on_ampl_error"] = "yes"

    @override
    def solve_single_step(self, env: EnvCore):
        # Solver 解决: 已知 config, 贪心得到当前奖励的最大值
        # actions: (N, A), max reward(N,A)
        # reward: closer + reach - collision - fear
        # target func: f(actions) = closer - fear + reach
        # 定义0-1指示变量，表示选中哪个方向
        target_pos: list[list[float]] = [
            env.agents[i].target_pos.tolist() for i in range(env.agent_num)
        ]
        current_pos: list[list[float]] = [
            env.agents[i].pos.tolist() for i in range(env.agent_num)
        ]
        velocities: list[list[float]] = [
            env.agents[i].velocity.tolist() for i in range(env.agent_num)
        ]
        closer_factor: float = env.tp.closer_factor
        reach_factor: float = env.tp.reach_factor
        scare_factor: float = env.tp.Scare_Factor

        humans = env.humans
        model = pyo.ConcreteModel()
        model.dimension = 3
        model.N = pyo.RangeSet(0, env.agent_num - 1)
        if not env.tp.action_discrete:
            # X变为连续变量，每个n对应dimension个连续变量，表示单位方向向量的三个分量
            model.X = pyo.Var(
                model.N, range(model.dimension), domain=pyo.Reals, bounds=(-1, 1)
            )

            # 添加约束，确保每个X[n]可达单位向量：x^2 + y^2 + z^2 <= 1(cube)
            def unit_vector_constraint(model, n):
                return sum(model.X[n, d] ** 2 for d in range(model.dimension)) <= 1

            model.unit_vector_constr = pyo.Constraint(
                model.N, rule=unit_vector_constraint
            )
        else:
            directions = {
                0: [1, 0, 0],
                1: [-1, 0, 0],
                2: [0, 1, 0],
                3: [0, -1, 0],
                4: [0, 0, 1],
                5: [0, 0, -1],
            }
            model.X_index = pyo.Var(model.N, domain=pyo.Integers, bounds=(0, 5))
            model.Y = pyo.Var(
                model.N, range(6), domain=pyo.Binary
            )  # 辅助变量, 只选择一个index

            def one_hot_rule(model, n):
                return sum(model.Y[n, k] for k in range(6)) == 1

            model.index_one_hot_constr = pyo.Constraint(model.N, rule=one_hot_rule)

            def X_rule(model, n, d):
                return sum(directions[k][d] * model.Y[n, k] for k in range(6))

            model.X = pyo.Expression(model.N, range(model.dimension), rule=X_rule)

        # 当前假设 current_pos 是一个二维数组或者类似结构，current_pos[n][d]为位置分量
        def next_pos_rule(model, n, d):
            return current_pos[n][d] + model.X[n, d]

        model.next_pos = pyo.Expression(
            model.N, range(model.dimension), rule=next_pos_rule
        )

        model.input_constraint = pyo.ConstraintList()
        num_humans = len(humans)
        human_positions = np.array([h.position for h in env.humans])
        human_velocities = np.array([h.velocity for h in env.humans])
        scare_factors = np.array([h.scare_factor for h in env.humans])
        cutoff_distance = env.tp.Cutoff_Scare_Distance

        def pyo_clip(model, x, a, b, M=1e5, name_prefix="clip"):
            """
            在模型中加入 y = clip(x, a, b) 约束，使用Big-M方法。
            clip定义： y = a if x < a; y = b if x > b; else y = x

            参数：
            model: Pyomo模型对象
            x: Pyomo Var变量（原始变量）
            a: 下界（数值）
            b: 上界（数值），需满足 a < b
            M: Big-M值，默认1e5，建议根据实际x的范围调整
            name_prefix: 变量和约束名前缀，避免重名

            返回：
            y: clip后的Pyomo Var变量
            binaries: 辅助二进制变量列表 [z_low, z_high]
            """

            # 新变量 y，取值范围限制为 [a, b]
            y = pyo.Var(bounds=(a, b))
            setattr(model, f"{name_prefix}_y", y)

            # 辅助二进制变量
            z_low = pyo.Var(domain=pyo.Binary)
            z_high = pyo.Var(domain=pyo.Binary)
            setattr(model, f"{name_prefix}_z_low", z_low)
            setattr(model, f"{name_prefix}_z_high", z_high)

            # 互斥约束，最多一个为1
            setattr(
                model,
                f"{name_prefix}_exclusive",
                pyo.Constraint(expr=z_low + z_high <= 1),
            )

            # 当z_low=1时，x <= a
            setattr(
                model,
                f"{name_prefix}_low_bound",
                pyo.Constraint(expr=x <= a + M * (1 - z_low)),
            )

            # 当z_high=1时，x >= b
            setattr(
                model,
                f"{name_prefix}_high_bound",
                pyo.Constraint(expr=x >= b - M * (1 - z_high)),
            )

            # 当z_low=1，y = a
            setattr(
                model,
                f"{name_prefix}_y_low_ub",
                pyo.Constraint(expr=y <= a + M * (1 - z_low)),
            )
            setattr(
                model,
                f"{name_prefix}_y_low_lb",
                pyo.Constraint(expr=y >= a - M * (1 - z_low)),
            )

            # 当z_high=1，y = b
            setattr(
                model,
                f"{name_prefix}_y_high_ub",
                pyo.Constraint(expr=y <= b + M * (1 - z_high)),
            )
            setattr(
                model,
                f"{name_prefix}_y_high_lb",
                pyo.Constraint(expr=y >= b - M * (1 - z_high)),
            )

            # 当z_low=0且z_high=0时，y = x
            # 通过不等式约束实现（允许误差被big-M控制）
            setattr(
                model,
                f"{name_prefix}_y_x_ub",
                pyo.Constraint(expr=y - x <= M * (z_low + z_high)),
            )
            setattr(
                model,
                f"{name_prefix}_y_x_lb",
                pyo.Constraint(expr=y - x >= -M * (z_low + z_high)),
            )
            return y, [z_low, z_high]

        def pyo_max(model, xs, M=1e6, name_prefix="max"):
            """
            添加max约束：
            z = max{x in xs}

            参数：
            model: Pyomo模型
            xs: 可迭代Pyomo变量或表达式列表
            M: Big-M值，选足够大
            name_prefix: 变量及约束名前缀

            返回：
            z: 表示max的Pyomo变量
            binaries: 二进制变量列表
            """
            n = len(xs)
            z = pyo.Var(domain=pyo.Reals)
            setattr(model, f"{name_prefix}_z", z)

            # 辅助二进制变量
            z_binaries = [pyo.Var(domain=pyo.Binary) for i in range(n)]
            for i, v in enumerate(z_binaries):
                setattr(model, f"{name_prefix}_bin_{i}", v)

            # 选择变量的二进制互斥约束：可以是多个可取，若有需要可改写成单选
            # 这里我们不强制互斥，允许多个满足
            # 若需要严格单选，添加约束 sum z_binaries == 1

            # 约束：z >= x[i]
            for i, x_i in enumerate(xs):
                model.add_component(
                    f"{name_prefix}_bound_{i}", pyo.Constraint(expr=z >= x_i)
                )

            # 约束：z <= x[i] + M*(1 - z_binaries[i])，确保z从所有x中取最大者
            for i, x_i in enumerate(xs):
                model.add_component(
                    f"{name_prefix}_ub_{i}",
                    pyo.Constraint(expr=z <= x_i + M * (1 - z_binaries[i])),
                )

            # 保证至少有一个binary为1，防止无效组合
            model.add_component(
                f"{name_prefix}_bin_sum", pyo.Constraint(expr=sum(z_binaries) >= 1)
            )

            return z, z_binaries

        def pyo_min(model, xs, M=1e6, name_prefix="min"):
            """
            添加min约束：
            z = min{x in xs}

            参数：
            model: Pyomo模型
            xs: 可迭代Pyomo变量或表达式列表
            M: Big-M值，选足够大
            name_prefix: 变量及约束名前缀

            返回：
            z: 表示min的Pyomo变量
            binaries: 二进制变量列表
            """
            n = len(xs)
            z = pyo.Var(domain=pyo.Reals)
            setattr(model, f"{name_prefix}_z", z)

            z_binaries = [pyo.Var(domain=pyo.Binary) for i in range(n)]
            for i, v in enumerate(z_binaries):
                setattr(model, f"{name_prefix}_bin_{i}", v)

            # 约束：z <= x[i]
            for i, x_i in enumerate(xs):
                model.add_component(
                    f"{name_prefix}_bound_{i}", pyo.Constraint(expr=z <= x_i)
                )

            # 约束：z >= x[i] - M*(1 - z_binaries[i])，确保z从所有x中取最小者
            for i, x_i in enumerate(xs):
                model.add_component(
                    f"{name_prefix}_lb_{i}",
                    pyo.Constraint(expr=z >= x_i - M * (1 - z_binaries[i])),
                )

            # 保证至少有一个binary为1
            model.add_component(
                f"{name_prefix}_bin_sum", pyo.Constraint(expr=sum(z_binaries) >= 1)
            )

            return z, z_binaries

        def closer(model: pyo.ConcreteModel):
            return (
                pyo.quicksum(
                    -pyo.sqrt(
                        pyo.quicksum(
                            (model.next_pos[n, d] - target_pos[n][d]) ** 2
                            for d in range(model.dimension)
                        )
                    )
                    + math.sqrt(
                        sum(
                            (current_pos[n][d] - target_pos[n][d]) ** 2
                            for d in range(model.dimension)
                        )
                    )
                    for n in model.N
                )
                * closer_factor
                / math.sqrt(3)
            )

        def reach(model: pyo.ConcreteModel):
            expr = 0
            for n in model.N:
                dist_sq = sum(
                    (model.next_pos[n, d] - target_pos[n][d]) ** 2
                    for d in range(model.dimension)
                )
                # 接近目标时，根据阈值eps给分，使用平滑函数替代布尔判断
                # 这里用 exp(-k * dist_sq) 来近似，k 调参控制衰减速度
                k = 1000
                expr += pyo.exp(-k * dist_sq)
            return expr * reach_factor

        def scare(model: pyo.ConcreteModel):
            # 2. 定义 Pyomo 表达式
            fearness_sum = 0.0
            for i, item in enumerate(zip(current_pos, velocities)):
                pos, v = item
                for j in range(num_humans):
                    # 2.1 距离计算
                    distance = pyo.sqrt(
                        pyo.quicksum(
                            (model.next_pos[i, k] - human_positions[j][k]) ** 2
                            for k in range(3)
                        )
                    )
                    # clip()
                    distance, _ = pyo_clip(
                        name_prefix=f"distance_{i}_{j}",
                        model=model,
                        x=distance,
                        a=0.1,
                        b=cutoff_distance,
                        M=1e3,
                    )

                    # 2.2 速度向量的 cosine 相似度
                    agent_velocity_norm = pyo.sqrt(
                        pyo.quicksum(v[k] ** 2 for k in range(3))
                    )
                    human_velocity_norm = pyo.sqrt(
                        pyo.quicksum(human_velocities[j][k] ** 2 for k in range(3))
                    )
                    agent_velocity_norm = pyo_max(
                        model=model,
                        xs=[agent_velocity_norm, 1e-2],
                        M=1e4,
                        name_prefix=f"agent_velocity_norm_{i}_{j}",
                    )[0]
                    human_velocity_norm = pyo_max(
                        model=model,
                        xs=[human_velocity_norm, 1e-4],
                        M=1e4,
                        name_prefix=f"human_velocity_norm_{i}_{j}",
                    )[0]  # 避免除0

                    dot_product = pyo.quicksum(
                        human_velocities[j][k] * v[k] for k in range(3)
                    )
                    cosine_v = dot_product / (human_velocity_norm * agent_velocity_norm)

                    # 2.3 相对位置和速度的 cosine 相似度
                    rel_pos = [pos[k] - human_positions[j][k] for k in range(3)]
                    rel_vel = [v[k] - human_velocities[j][k] for k in range(3)]
                    rel_pos_norm = pyo_max(
                        model,
                        [
                            pyo.sqrt(pyo.quicksum(rel_pos[k] ** 2 for k in range(3))),
                            1e-4,
                        ],
                        M=1e4,
                        name_prefix=f"rel_pos_norm_{i}_{j}",
                    )[0]  # 避免除0
                    cosine_r = pyo.quicksum(
                        rel_vel[k] * rel_pos[k] for k in range(3)
                    ) / (1e-2 + agent_velocity_norm * rel_pos_norm)
                    pos_angle_factor = pyo_max(
                        model=model,
                        xs=[0, cosine_r],
                        M=1e3,
                        name_prefix=f"pos_angle_factor_{i}_{j}",
                    )[0]

                    # 2.4 计算 fearness
                    fearness = (1 / distance) * scare_factors[j] * (1 - cosine_v)

                    fearness_sum += fearness

            return fearness_sum * pos_angle_factor

        # 不碰约束
        model.no_collision = pyo.ConstraintList()
        for n in model.N:
            for m in model.N:
                if n < m:
                    # 计算两个agent的下一个位置
                    # 添加约束，确保两个agent的下一个位置不重叠
                    model.no_collision.add(
                        pyo.quicksum(
                            (model.next_pos[n, k] - model.next_pos[m, k]) ** 2
                            for k in range(model.dimension)
                        )
                        >= math.sqrt(1)  # 确保距离大于sqrt(3)?
                    )
        # 不碰建筑的约束
        model.no_building_collision = pyo.ConstraintList()
        buildings = env.buildings
        """
        cal_distance
        x_min, y_min, z_min, x_max, y_max, z_max = cube_param
        x, y, z = pos
        # 计算每个轴上的距离
        dx = max(x_min - x, 0, x - x_max)
        dy = max(y_min - y, 0, y - y_max)
        dz = max(z_min - z, 0, z - z_max)
        # 返回总的距离
        if dx <= 0 and dy <= 0 and dz <= 0:
            return 0.0
        return np.sqrt(dx**2 + dy**2 + dz**2)
        """
        for n in model.N:
            for i, building in enumerate(buildings):
                # distance to each building > 0.5
                building_param = building.to_array()
                # 计算下一个位置到建筑的距离
                distance = 1e-2
                for d in range(model.dimension):
                    delta = pyo_max(
                        model=model,
                        xs=[
                            building_param[d] - model.next_pos[n, d],
                            0,
                            -building_param[d + 3] + model.next_pos[n, d],
                        ],
                        M=1e3,
                        name_prefix=f"building_distance_{n}_{i}_{d}",
                    )[0]
                    distance += delta**2

                # 添加约束，确保距离大于0.5
                model.no_building_collision.add(pyo.sqrt(distance) >= 0.5)

        model.objective = pyo.Objective(
            expr=closer(model) - scare(model) + reach(model),
            sense=pyo.maximize,
        )
        print_model_info(model)
        results = self.solver.solve(model, tee=True)
        objective = pyo.value(model.objective)
        steps = np.array(
            [value(model.X[n, d]) for n in model.N for d in range(model.dimension)]
        ).reshape((len(model.N), model.dimension))

        success = (
            results.solver.termination_condition == pyo.TerminationCondition.optimal
            or results.solver.termination_condition == pyo.TerminationCondition.feasible
        )

        return objective, steps, success


def load_args():
    algo_name = "mappo"
    env_name = "pharos_discrete"
    algo_args, env_args = get_defaults_yaml_args(algo_name, env_name)
    return env_args


def get_env_args(N_Agents=10, N_Humans=8, action_discrete=False):
    # Example usage of PyomoSolver
    args = load_args()
    args["N_Agents"] = N_Agents  # Example number of agents
    args["N_Humans"] = N_Humans  # Example number of humans
    args["action_discrete"] = action_discrete  # Example action type
    return args


def test_pyomosolver():
    args = get_env_args()
    env = EnvCore(args=args)  # Assuming EnvCore is properly initialized
    solver = PyomoSolver(name="ipopt")

    obj, steps = solver.solve_single_step(env)
    print(f"Reward: {obj}")
    print(f"Steps taken: {steps}")
    print("PyomoSolver executed successfully.")


def update_humans(env: EnvCore, steps: np.ndarray):
    for i in range(env.human_num):
        env.humans[i].update(delta_t=1)
        # 如果走出边界, 折返
        if any(env.humans[i].position > np.array(env.tp.World_Max_Bound_Cube)) or any(
            env.humans[i].position < np.array(env.tp.World_Min_Bound_Cube)
        ):
            env.humans[i].position = np.clip(
                env.humans[i].position,
                np.array(env.tp.World_Min_Bound_Cube),
                np.array(env.tp.World_Max_Bound_Cube),
            )
            env.humans[i].velocity = -env.humans[i].velocity


def diff_solvers(solvers: dict[str, GenericSolver], env: EnvCore, timesteps: int = 50):
    """
    Compare the actions taken by the RL solver and the Pyomo solver.
    """
    envs = [EnvCore(args=get_env_args()) for _ in range(len(solvers))]
    for idx, (name, solver) in enumerate(solvers.items()):
        env = envs[idx]
        print(colored(f"Running solver: {name}", "green"))
        rewards = []
        # TODO: 绘制3d可视化图(保存Vis格式json)
        start_time = time.time()
        positions = [[] for _ in range(env.agent_num)]
        end_timestamp = 0
        for t in range(timesteps):
            reward, steps, success = solver.solve_single_step(env)
            if not success:
                print(colored(f"Solver {name} failed at timestep {t}.", "red"))
                end_timestamp = t
                break
            rewards.append(reward)
            for i in range(env.agent_num):
                env.agents[i].pos = env.agents[i].pos.astype(np.float64)
                env.agents[i].velocity = env.agents[i].velocity.astype(np.float64)
                env.agents[i].pos += steps[i]
                env.agents[i].velocity = steps[i]
                positions[i].append(env.agents[i].pos.copy())
                print(
                    f"Agent {i} Position: {env.agents[i].pos}, Velocity: {env.agents[i].velocity}"
                )

            update_humans(env, steps)

        # 生成报告
        with open(f"{name}_report.txt", "w") as f:
            f.write(f"Solver: {name}\n")
            f.write(f"Total Reward: {sum(rewards)}\n")
            f.write(f"Average Reward: {np.mean(rewards)}\n")
            f.write("Positions:\n")
            if end_timestamp > 0:
                f.write(
                    f"Avg Time: {(time.time() - start_time) / end_timestamp:.2f} seconds\n"
                )
            for i, pos in enumerate(positions):
                f.write(f"Agent {i}: {pos}\n")
        if end_timestamp > 0:
            print(
                colored(
                    f"Avg Time: {(time.time() - start_time) / end_timestamp:.2f} seconds\n"
                ),
                "yellow",
            )
        print(colored(f"Total Reward: {sum(rewards)}"), "yellow")


class MLPWithLayerNormSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(24, 64, bias=True)
        self.norm1 = nn.LayerNorm(64)
        self.act1 = nn.SiLU()

        self.layer2 = nn.Linear(64, 64, bias=True)
        self.norm2 = nn.LayerNorm(64)
        self.act2 = nn.SiLU()

        self.output_layer = nn.Linear(64, 7, bias=True)
        self.norm_out = nn.LayerNorm(7)
        self.softmax = nn.Softmax(dim=-1)

        # 权重初始化：Xavier均匀初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.layer2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.output_layer(x)
        x = self.norm_out(x)
        x = self.softmax(x)

        return x


def test_inference_speed(time_limit: float = 10.0):
    """
    Test the inference speed of the RL solver.
    """
    model = MLPWithLayerNormSiLU()
    start = time.time()
    cnt = 0
    while time.time() - start < time_limit:
        x = torch.randn(1, 24)
        output = model(x)
        cnt += 1
    print(f"Inference speed: {cnt / (time.time() - start):.2f} FPS")


if __name__ == "__main__":
    # 1. load RL models
    # 2. begin inference
    # 3. record metrics
    # 4. solver solve
    # 5. record metrics
    # 6. diff
    # HINT: 离散很容易解出来无解

    # diff_solvers(
    #     {
    #         "ipopt": PyomoSolver(name="ipopt"),
    #     },
    #     env=EnvCore(get_env_args()),
    #     timesteps=50,
    # )
    test_inference_speed()
