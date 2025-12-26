# 计算熵值, 1. 读取json数据 2.统计不同格点的占用率 3.计算熵值 4.对于多组数据, 绘图查看episode-熵值与episode-奖励的关系

from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
import tensorboard as tb
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import argparse


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


WorldSize = (30, 10, 30)


def calculate_entropy(data: list[AgentVis]) -> float:
    """Calculate entropy based on the distribution of agents in the world."""
    grid_size = 1.0  # Define the size of each grid cell
    grid_counts = {}
    agents = len(set([agent.uid for agent in data]))
    total_timesteps = len(data) / agents
    for agent in data:
        # Calculate grid cell index
        x_index = int(agent.position[0] // grid_size)
        y_index = int(agent.position[1] // grid_size)
        z_index = int(agent.position[2] // grid_size)

        # Create a tuple for the grid cell
        grid_cell = (x_index, y_index, z_index)
        # Count agents in each grid cell
        if grid_cell not in grid_counts:
            grid_counts[grid_cell] = 0
        grid_counts[grid_cell] += 1

    # Calculate probabilities and entropy
    # p: the probability of each grid cell being occupied
    probabilities = [count / total_timesteps for count in grid_counts.values()]

    entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
    return entropy


# 读取base_dir目录下的所有json文件
def read_json_files(base_dir: str) -> list[AgentVis]:
    data = defaultdict(list)
    for filename in os.listdir(base_dir):
        if filename.endswith(".json"):
            episodes = int(filename.split("_")[1])
            with open(os.path.join(base_dir, filename), "r") as f:
                json_data = json.load(f)
                for device in json_data.get("devices", []):
                    data[episodes].append(AgentVis(**device))
    return data


def tensorboard_to_dataframe(path):
    """将 TensorBoard 数据转换为 pandas DataFrame."""
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


def draw_picture(
    vis_dir: str, rewards_data_file: str, output_file: str = "entropy.png"
):
    # 读取数据
    data = read_json_files(vis_dir)

    for ep, agents in data.items():
        entropy = calculate_entropy(agents)
        print(f"Episode {ep}: Entropy = {entropy:.4f}")

    # 绘图
    import matplotlib.pyplot as plt

    episodes = list(data.keys())
    entropies = [calculate_entropy(agents) for agents in data.values()]
    rewards_pd = tensorboard_to_dataframe(rewards_data_file)

    # Debug: 打印DataFrame的列名和前几行数据
    print("DataFrame columns:", rewards_pd.columns.tolist())
    print("DataFrame head:")
    print(rewards_pd.head())

    # 检查是否有'value'列，如果没有则尝试其他可能的列名
    if "value" in rewards_pd.columns:
        rewards = rewards_pd["value"].tolist()
    else:
        print("Available columns:", rewards_pd.columns.tolist())
        # 如果没有value列，我们需要找到包含奖励数据的列
        # 通常可能是某个特定的tag
        if len(rewards_pd) > 0:
            # 尝试找到包含奖励相关信息的tag
            reward_tags = [
                tag for tag in rewards_pd["tag"].unique() if "reward" in tag.lower()
            ]
            if reward_tags:
                print(f"Found reward tags: {reward_tags}")
                # 使用第一个找到的奖励tag
                reward_data = rewards_pd[rewards_pd["tag"] == reward_tags[0]]
                rewards = reward_data["value"].tolist()
            else:
                # 如果没有找到明确的奖励tag，使用所有数据的value列
                rewards = rewards_pd["value"].tolist()
        else:
            print("No data in TensorBoard file")
            rewards = []
    # episode和entropy按episodes排序
    episodes, entropies = zip(*sorted(zip(episodes, entropies)))

    # 检查是否有奖励数据
    if len(rewards) == 0:
        print("No reward data found, showing only entropy plot")
        plt.plot(episodes, entropies, marker="o", label="Entropy", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Entropy")
        plt.title("Episode vs Entropy")
        plt.legend()
        plt.grid()
        plt.savefig("entropy.png")
        return

    # 检查数据长度是否匹配
    if len(rewards) != len(episodes):
        print(
            f"Warning: Rewards data length ({len(rewards)}) doesn't match episodes length ({len(episodes)})"
        )
        # 截取到较短的长度
        min_len = min(len(rewards), len(episodes), len(entropies))
        rewards = rewards[:min_len]
        episodes = list(episodes[:min_len])
        entropies = list(entropies[:min_len])

    # 将reward缩放到0,1
    # 将entropies缩放到0,1
    rewards = np.array(rewards)
    entropies = np.array(entropies)

    if np.max(rewards) != np.min(rewards):
        rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards))
    else:
        rewards = np.ones_like(rewards) * 0.5  # 如果所有奖励都相同，设为0.5

    if np.max(entropies) != np.min(entropies):
        entropies = (entropies - np.min(entropies)) / (
            np.max(entropies) - np.min(entropies)
        )
    else:
        entropies = np.ones_like(entropies) * 0.5  # 如果所有熵值都相同，设为0.5
    # 计算pearson相关系数
    correlation = np.corrcoef(entropies, rewards)[0, 1]
    print(f"Pearson correlation between entropy and rewards: {correlation:.4f}")
    plt.plot(
        episodes, entropies, marker="o", label="Entropy", color="blue"
    )  # 添加熵值曲线
    plt.plot(
        episodes, rewards, marker="x", label="Rewards", color="orange"
    )  # 添加奖励曲线
    # 设置为不同的颜色
    plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("Episode vs Entropy")
    plt.grid()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate entropy and analyze relationship with rewards"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Base directory containing vis and logs folders, e.g. \
            results/pharos_discrete/pharos_discrete/mappo/installtest/seed-00001-2025-06-27-04-55-44",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="entropy.png",
        help="Output file name for the plot",
    )

    args = parser.parse_args()

    vis_dir = f"{args.base_dir}/vis"
    rewards_dir = Path(f"{args.base_dir}/logs/eval_average_episode_rewards")

    rewards_data_file = str(list(rewards_dir.rglob("*.ps"))[0])
    print(f"Rewards data file: {rewards_data_file}")

    draw_picture(
        vis_dir=vis_dir,
        rewards_data_file=rewards_data_file,
        output_file=args.output_file,
    )
