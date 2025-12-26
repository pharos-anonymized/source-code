# 计算熵值, 1. 读取json数据 2.统计不同格点的占用率 3.计算熵值 4.对于多组数据, 绘图查看episode-熵值与episode-奖励的关系

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, validator
import tensorboard as tb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot as plt
import argparse

"""
{
  "devices": [
    {
      "uid": "vehicle/10001",
      "position": [24.0, 48.0, 26.0],
      "velocity": [0.0, 0.0, 0.0],
      "ts": 1742393414649,
      "include_area": [22.0, 46.0, 24.0, 26.0, 50.0, 28.0],
    },
    {
      "uid": "vehicle/10002",
      "position": [31.0, 25.0, 50.0],
      "velocity": [0.0, 0.0, 0.0],
      "ts": 1742393414649,
      "include_area": [29.0, 23.0, 48.0, 33.0, 27.0, 52.0],
    },
    // ...
  ],
  "humans": [
    {
      "hid": "human/10001",
      "position": [24.0, 1.0, 26.0],
      "velocity": [1.0, 0.0, 0.0],
      "ts": 1742393414649,
    },
    // ...
  ],
  "buildings": [
      {
        "id": "b01"
        "bbox": [1,2,3,4,5,6],
      }
  ] 
}

"""
base_dir = "vis"


class AgentVis(BaseModel):
    uid: str
    position: list[float] = Field(..., min_items=3, max_items=3)
    velocity: list[float] = Field(..., min_items=3, max_items=3)
    ts: int
    include_area: list[float] = Field(..., min_items=6, max_items=6)
    target_pos: list[float] = None


class HumanVis(BaseModel):
    hid: str
    position: list[float] = Field(..., min_items=3, max_items=3)
    velocity: list[float] = Field(..., min_items=3, max_items=3)
    ts: int


class BuildingVis(BaseModel):
    id: str | int
    bbox: list[float] = Field(..., min_items=6, max_items=6)


WorldSize = (30, 10, 30)


def find_tb_ps_file(base_dir: str):
    p = Path(base_dir)
    ps_files = list(p.glob("**/*.ps"))
    assert len(ps_files) == 1, f"{ps_files} found in {base_dir}"
    return ps_files[0]


def tensorboard_to_dataframe(path: str | Path):
    """将 TensorBoard 数据转换为 pandas DataFrame."""
    if isinstance(path, Path):
        path = str(path)
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    data = []
    tags = ea.scalars.Keys()

    for tag in tags:
        for event in ea.scalars.Items(tag):
            data.append(
                {
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                    "wall_time": event.wall_time,
                }
            )

    df = pd.DataFrame(data)
    return df


# 读取base_dir目录下的所有json文件
def read_vis_json_files(base_dir: str) -> list[AgentVis]:
    base_dir = Path(base_dir) / "vis"
    data = defaultdict(list)
    for filename in os.listdir(base_dir):
        if filename.endswith(".json"):
            episodes = int(filename.split("_")[1])
            with open(os.path.join(base_dir, filename), "r") as f:
                json_data = json.load(f)
                for device in json_data.get("devices", []):
                    data[episodes].append(AgentVis(**device))
                for human in json_data.get("humans", []):
                    data[episodes].append(HumanVis(**human))
                for building in json_data.get("buildings", []):
                    data[episodes].append(BuildingVis(**building))
    return data


"""
- [] 只考虑无人机的时候，不考虑人和建筑物以及无人机的大小，测试在不同无人机数量下，连续和离散空间性能的变化，收敛和训练情况对比
- [] 当前离散空间跟求解器对比最终求解的奖励，求解具体一个时刻的奖励，奖励要接近求解器（可以增加不同无人机数量的情况下的奖励对比）
"""


def load_discrete_data(base_dirs: list[str]) -> list:
    pass


class CriticTableNames(Enum):
    AvgStepReward = "average_step_rewards"
    CriticGradNorm = "critic_grad_norm"
    ValueLoss = "value_loss"


class ExtraTableNames(Enum):
    EvalAvgA2ACollisions = "eval_avg_a2a_collisionss"
    EvalAvgA2BCollisions = "eval_avg_a2b_collisionss"
    EvalAvgA2HCollisions = "eval_avg_a2h_collisionss"
    EvalAvgFearPenaltys = "eval_avg_fear_penaltys"
    EvalAvgCollisionPenaltys = "eval_avg_collision_penaltys"
    EvalAvgArriveReward = "eval_avg_arrive_rewards"
    EvalAvgTargetCloserReward = "eval_avg_target_closer_rewards"
    EvalMaxEpisodeRewards = "eval_max_episode_rewards"
    TrainEpisodeRewards = "train_episode_rewards"


class ActorTableNames(Enum):
    ActorGradNorm = "actor_grad_norm"
    DistEntropy = "dist_entropy"
    PolicyLoss = "policy_loss"
    Ratio = "ratio"


TableName = CriticTableNames | ExtraTableNames | ActorTableNames


def table2pd(base_dir: str, table_name: TableName):
    """
    将 tensorboard 中的 table 转换为 pandas DataFrame
    """
    base_p = Path(base_dir)
    logs_p = base_p / "logs"
    with open(base_p / "config.json", "r") as f:
        config_dict = json.load(f)
    env_arg = config_dict["env_args"]
    n_agents = env_arg["N_Agents"]
    logs_p
    if table_name in CriticTableNames:
        p = logs_p / "critic" / table_name.value
        ps_file = find_tb_ps_file(p)
        return tensorboard_to_dataframe(ps_file)
    elif table_name in ActorTableNames:
        df_list = []
        for i in range(n_agents):
            p = logs_p / f"agent{i}" / table_name.value
            ps_file = find_tb_ps_file(p)
            df_list.append(tensorboard_to_dataframe(ps_file))
        # 求均值
        df = pd.concat(df_list, axis=0)
        df = (
            df.groupby("step")
            .agg(
                {
                    "value": "mean",  # value 列求均值
                    "tag": "first",  # tag 列取第一个（默认所有同 step 应该相同）
                    "wall_time": "first",  # wall_time 也取第一个
                }
            )
            .reset_index()
        )
        return df
    return tensorboard_to_dataframe(find_tb_ps_file(logs_p / table_name.value))


def draw_smooth_picture(
    pd: pd.DataFrame, ax=None, label=None, color=None, normalize=False, alpha=0.15
):
    """
    绘制平滑曲线 + 半透明误差区间
    :param pd: pandas DataFrame, 包含 step 和 value 列
    :param ax: matplotlib Axes, 可选
    :param label: 曲线标签, 可选
    :param color: 曲线颜色, 可选
    :param alpha: 阴影透明度
    :return: 最后收敛均值
    """
    if ax is None:
        fig, ax = plt.subplots()

    # 计算滚动均值和滚动标准差
    if not normalize:
        rolling_mean = pd["value"].rolling(window=10, min_periods=1).mean()
        rolling_std = pd["value"].rolling(window=10, min_periods=1).std()
    else:
        v_min = pd["value"].min()
        v_max = pd["value"].max()
        pd["value_norm"] = (pd["value"] - v_min) / (v_max - v_min)
        rolling_mean = pd["value_norm"].rolling(window=10, min_periods=1).mean()
        rolling_std = pd["value_norm"].rolling(window=10, min_periods=1).std()

    # 画均值曲线
    ax.plot(pd["step"], rolling_mean, label=label, color=color)

    # 画均值±滚动标准差的阴影区域，alpha设置透明度
    ax.fill_between(
        pd["step"],
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color=color,
        alpha=alpha,
    )

    if label:
        ax.legend()


def avg_agent_rewards_diff_agents(agent_dir_mapping: dict[int, str]):
    """
    不同数量的无人机的实验, 相同episode时, 每架无人机平均reward的对比图, 每一个数量都是一条曲线
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (n_agents, base_dir) in enumerate(agent_dir_mapping.items()):
        # 读取每个无人机数量的平均奖励数据
        pd_reward = table2pd(base_dir, CriticTableNames.AvgStepReward)
        # 步长归一化
        pd_reward["value"] = pd_reward["value"] / n_agents
        pd_reward["value"] /= 10
        # 绘制曲线，并获取最后收敛值
        draw_smooth_picture(
            pd_reward, ax=ax, label=f"N={n_agents}", color=color_a30[idx], alpha=0.15
        )
    ax.set_title("Average Step Reward by Number of Agents")
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Reward")
    ax.legend()
    ax.grid(True)
    # # 设置Y轴范围，聚焦收敛区间
    ax.set_ylim(-2, 1)
    return fig, ax


def avg_agent_rewards_diff_label(label_dir_mapping: dict[str, str], label_name: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (label, base_dir) in enumerate(label_dir_mapping.items()):
        # 读取每个无人机数量的平均奖励数据
        pd_reward = table2pd(base_dir, CriticTableNames.AvgStepReward)
        # 步长归一化
        pd_reward["value"] = pd_reward["value"]
        pd_reward["value"] /= 10
        # 绘制曲线，并获取最后收敛值
        draw_smooth_picture(
            pd_reward, ax=ax, label=f"{label}", color=color_a30[idx], alpha=0.15
        )
    ax.set_title(f"Step Reward by different {label_name}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)
    # ax.set_ylim(-20, 10)
    return fig, ax


def colormap(value: np.ndarray):
    """
    将一系列数值映射到高对比度颜色
    :param value: 数值
    :return: RGB颜色元组
    """
    # 使用高对比度的颜色列表
    high_contrast_colors = [
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#800000",  # Maroon
        "#808000",  # Olive
        "#008080",  # Teal
        "#800080",  # Purple
        "#FF7F00",  # Orange
        "#7F00FF",  # Violet
    ]
    unique_values = np.unique(value)
    return [
        high_contrast_colors[i % len(high_contrast_colors)]
        for i in range(len(unique_values))
    ]


color_a30 = colormap(np.array([4, 6, 8, 10, 15, 20, 30]))


def experiment_subgraphs(base_dir):
    # 读取数据
    # vis_data = read_vis_json_files(base_dir)
    # 假设你有6组数据和对应标签、颜色
    pd_reward = table2pd(base_dir, CriticTableNames.AvgStepReward)
    pd_entropy = table2pd(base_dir, ActorTableNames.DistEntropy)
    pd_fear = table2pd(base_dir, ExtraTableNames.EvalAvgFearPenaltys)
    # pd_collision = table2pd(base_dir, ExtraTableNames.EvalAvgCollisionPenaltys)
    # pd_arrive = table2pd(base_dir, ExtraTableNames.EvalAvgArriveReward)
    pd_target_closer = table2pd(base_dir, ExtraTableNames.EvalAvgTargetCloserReward)
    pd_a2a_collisions = table2pd(base_dir, ExtraTableNames.EvalAvgA2ACollisions)
    data_groups = [
        (pd_reward, "AvgStepReward", "blue"),
        (pd_entropy, "DistEntropy", "orange"),
        (pd_fear, "EvalAvgFearPenaltys", "green"),
        # (pd_collision, "EvalAvgCollisionPenaltys", "red"),
        # (pd_arrive, "EvalAvgArriveReward", "purple"),
        (pd_target_closer, "EvalAvgTargetCloserReward", "brown"),
        (pd_a2a_collisions, "EvalAvgA2ACollisions", "cyan"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))  # 2行3列子图，调整大小
    axes = axes.flatten()  # 将二维数组拉平成1维方便迭代

    for ax, (df, label, color) in zip(axes, data_groups):
        draw_smooth_picture(df, ax=ax, label=label, color=color)
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Pharos Metrics")  # 总标题
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # 调整子图间距，避免标题覆盖
    return fig, axes


def experiment_diff_subgraph(
    exps: dict[str, str],
    table_name: TableName,
    normalize_table_name: Optional[TableName] = None,
):
    """
    绘制不同实验目录下的表格数据对比图
    :param exps: 一个字典，键为实验名称，值为实验目录
    :param table_name: 表格名称
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (name, base_dir) in enumerate(exps.items()):
        pd_data = table2pd(base_dir, table_name)
        if normalize_table_name is not None:
            norm_data = table2pd(base_dir, normalize_table_name)["value"]
            eps = 1e-3
            # 归一化norm_data
            norm_data = norm_data / (norm_data.max() - norm_data.min())
            norm_data = norm_data + eps  # 避免除0错误
            pd_data["value"] = pd_data["value"] / norm_data
        draw_smooth_picture(
            pd_data,
            ax=ax,
            label=f"{name}"
            if not normalize_table_name
            else f"{name} (normalized by {normalize_table_name.value})",
            color=color_a30[idx % len(color_a30)],
            alpha=0.15,
        )
    ax.set_title(f"{table_name.value} Comparison")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    return fig, ax


class ExperimentType(Enum):
    DiscreteAction = "discrete"


def recursive_find(key: str, args: dict):
    if key in args:
        return args[key]
    for k, v in args.items():
        if isinstance(v, dict):
            result = recursive_find(key, v)
            if result is not None:
                return result
    return None


def find_experiment_dir(results_dir: str, filter: dict[str, Any]) -> Optional[Path]:
    matches = []

    for subdir in Path(results_dir).iterdir():
        config = json.load(open(subdir / "config.json"))
        if all(recursive_find(k, config) == v for k, v in filter.items()):
            datetime_str = "-".join(subdir.name.removeprefix("seed-").split("-")[1:])
            matches.append((datetime_str, subdir))
    if len(matches) == 0:
        return None
    # 多个时, 选取最新的实验
    return sorted(matches, key=lambda x: x[0], reverse=True)[0][1]


n_agents_filter_key = "N_Agents"
num_env_steps_filter_key = "num_env_steps"
action_filter_key = "activation_func"
continue_train_key = "model_dir"
has_human_filter_key = "enable_human_fear"
has_building_filter_key = "enable_buildings"


def exp_avg_agent_rewards_diff_agents(
    results_dir: str,
    extra_filter: dict[str, Any],
    agent_nums: list[int] = [4, 6, 8, 10, 15, 20, 30],
    filename: str = "avg_agent_rewards_diff_agents.png",
):
    """
    绘制不同数量无人机的平均奖励对比图
    """
    agent_dir_mapping = {}
    for agent_num in agent_nums:
        agent_dir_mapping[agent_num] = str(
            find_experiment_dir(
                results_dir,
                extra_filter | {n_agents_filter_key: agent_num},
            )
        )
    fig, ax = avg_agent_rewards_diff_agents(agent_dir_mapping)
    plt.savefig(filename)


def exp_avg_agent_rewards_diff_algos(
    algo_mapping: dict[str, str],
    extra_filter: dict[str, Any],
    filename: str = "avg_agent_rewards_diff_algos.png",
    title: str = "Rewards by Algorithms",
):
    """
    绘制不同算法下的平均奖励对比图
    :param: algo_mapping: 一个字典，键为算法名称，值为对应的实验目录(子目录为seed-<seed_value>)
    """
    algo_dir_mapping = {}
    for algo_name, base_dir in algo_mapping.items():
        algo_dir_mapping[algo_name] = str(
            find_experiment_dir(
                results_dir=base_dir,
                filter=extra_filter,
            )
        )
        print(f"Algo {algo_name} dir: {algo_dir_mapping[algo_name]}")
    print(f"Algo dir mapping: {algo_dir_mapping}")
    fig, ax = avg_agent_rewards_diff_label(algo_dir_mapping, label_name="algorithms")
    ax.set_title(title)
    plt.savefig(filename)


def exp_subgraphs(
    results_dir: str,
    extra_filter: dict[str, Any],
    filename: str = "experiment_subgraphs.png",
):
    """
    绘制实验结果的子图
    """
    base_dir = find_experiment_dir(results_dir, extra_filter)
    print(f"Base dir found: {base_dir}")
    if base_dir:
        fig, axes = experiment_subgraphs(base_dir=str(base_dir))
        plt.savefig(filename)
    else:
        print(f"No experiment found with filters: {extra_filter}")


def calulate_metric_avg(
    base_dir: str,
    start_step: int,
    end_step: int,
    metric: CriticTableNames | ActorTableNames | ExtraTableNames,
) -> float:
    df = table2pd(base_dir, metric)
    df_filtered = df[(df["step"] >= start_step) & (df["step"] <= end_step)]
    print(f"Steps filtered: {df_filtered['step'].tolist()}")
    if df_filtered.empty:
        raise ValueError(f"No data found for {metric.value} in the specified range.")
    return df_filtered["value"].mean()


# 你可以指定 pivot=True 得到宽表
# df_wide = experiment.get_scalars(pivot=True)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate various experimental plots")
    parser.add_argument(
        "--plot-type",
        type=str,
        required=True,
        choices=[
            "agent-rewards",
            "subgraphs",
            "algo-comparison",
            "fear-penalty",
            "avg-agent-metrics",
        ],
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--expname", type=str, default="installtest", help="Experiment name"
    )
    parser.add_argument("--algo", type=str, default="mappo", help="Algorithm name")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--discrete-env",
        type=str,
        default="pharos_discrete",
        help="Discrete environment name",
    )
    parser.add_argument(
        "--agent-nums",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 15, 20, 30],
        help="Number of agents to test",
    )
    parser.add_argument(
        "--has-human", action="store_true", help="Include human fear penalty"
    )
    parser.add_argument(
        "--has-building", action="store_true", help="Include building constraints"
    )
    parser.add_argument("--filename", type=str, help="Output filename (optional)")
    parser.add_argument(
        "--metric-name",
        type=str,
        default="EvalA2ACollisions",
        help="Metric name to calculate average (optional, for avg-agent-metrics plot)",
    )
    args = parser.parse_args()

    # Base filter configuration
    base_filter = {
        num_env_steps_filter_key: 5000000,
        continue_train_key: None,
    }
    if args.has_human:
        base_filter[has_human_filter_key] = True
    else:
        base_filter[has_human_filter_key] = False
    if args.has_building:
        base_filter[has_building_filter_key] = True
    else:
        base_filter[has_building_filter_key] = False

    discrete_dir = os.path.join(
        args.results_dir, args.discrete_env, args.discrete_env, args.algo, args.expname
    )
    print(discrete_dir)

    if args.plot_type == "agent-rewards":
        filename = args.filename or "avg_agent_rewards_diff_agents.png"
        exp_avg_agent_rewards_diff_agents(
            results_dir=discrete_dir,
            extra_filter=base_filter,
            agent_nums=args.agent_nums,
            filename=filename,
        )

    elif args.plot_type == "subgraphs":
        filename = args.filename or "experiment_subgraphs.png"
        exp_subgraphs(
            results_dir=discrete_dir,
            extra_filter=base_filter | {n_agents_filter_key: args.agent_nums[0]},
            filename=filename,
        )

    elif args.plot_type == "algo-comparison":
        algos = ["mappo", "hatrpo", "happo"]
        algo_dir_mapping = {}
        for algo_name in algos:
            algo_discrete_dir = os.path.join(
                args.results_dir,
                args.discrete_env,
                args.discrete_env,
                algo_name,
                args.expname,
            )
            algo_dir_mapping[algo_name] = algo_discrete_dir

        for agent_num in args.agent_nums:
            filter_with_agents = base_filter | {n_agents_filter_key: agent_num}
            filename = args.filename or f"avg_agent_rewards_diff_algos_{agent_num}A.png"
            exp_avg_agent_rewards_diff_algos(
                algo_mapping=algo_dir_mapping,
                extra_filter=filter_with_agents,
                filename=filename,
                title=f"Rewards by Algorithms (N={agent_num} Agents)",
            )

    elif args.plot_type == "fear-penalty":
        algos = ["mappo", "hatrpo", "happo"]
        exps = {}
        for algo_name in algos:
            algo_discrete_dir = os.path.join(
                args.results_dir,
                args.discrete_env,
                args.discrete_env,
                algo_name,
                args.expname,
            )
            exp_dir = find_experiment_dir(
                algo_discrete_dir,
                base_filter | {n_agents_filter_key: args.agent_nums[0]},
            )
            if exp_dir:
                exps[algo_name] = str(exp_dir)

        if exps:
            fig, ax = experiment_diff_subgraph(
                exps=exps,
                table_name=ExtraTableNames.EvalAvgFearPenaltys,
                normalize_table_name=ExtraTableNames.EvalMaxEpisodeRewards,
            )
            filename = args.filename or "avg_fear_penaltys_diff_algo.png"
            plt.savefig(filename)
    elif args.plot_type == "avg-agent-metrics":
        metric_name = args.metric_name
        if metric_name == "EvalA2ACollisions":
            metric = ExtraTableNames.EvalAvgA2ACollisions
        elif metric_name == "EvalA2BCollisions":
            metric = ExtraTableNames.EvalAvgA2BCollisions
        elif metric_name == "EvalA2HCollisions":
            metric = ExtraTableNames.EvalAvgA2HCollisions
        elif metric_name == "EvalAvgFearPenaltys":
            metric = ExtraTableNames.EvalAvgFearPenaltys
        elif metric_name == "EvalAvgCollisionPenaltys":
            metric = ExtraTableNames.EvalAvgCollisionPenaltys
        elif metric_name == "EvalAvgArriveReward":
            metric = ExtraTableNames.EvalAvgArriveReward
        elif metric_name == "EvalAvgTargetCloserReward":
            metric = ExtraTableNames.EvalAvgTargetCloserReward
        elif metric_name == "EvalMaxEpisodeRewards":
            metric = ExtraTableNames.EvalMaxEpisodeRewards
        elif metric_name == "TrainEpisodeRewards":
            metric = ExtraTableNames.TrainEpisodeRewards
        elif metric_name == "AvgStepReward":
            metric = CriticTableNames.AvgStepReward
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
        if not args.results_dir:
            raise ValueError("results_dir must be specified for avg-agent-metrics plot")
        start_step = 4000000
        end_step = 5000000
        avg_value = calulate_metric_avg(
            base_dir=args.results_dir,
            start_step=start_step,
            end_step=end_step,
            metric=metric,
        )
        print(
            f"Average {metric.value} from step {start_step} to {end_step}: {avg_value}"
        )
