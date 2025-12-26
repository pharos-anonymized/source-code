import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from core.env import Env, RewardRecord, calc_reward, env_step
from core.log_data import LogData
from core.utils import seed_everything


def calculate_stats(reward_records: list[RewardRecord]) -> dict[str, float]:
    """Calculate reward and time statistics from reward records."""
    rewards = [r.reward for r in reward_records]
    times = [r.time_taken for r in reward_records if r.time_taken is not None]

    return {
        "total_reward": float(sum(rewards)),
        "average_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "time_mean": float(np.mean(times)),
        "time_std": float(np.std(times)),
    }


def save_results(log_data: LogData, reward_records: list[RewardRecord], output_dir: Path) -> None:
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "vis_data.json", "w") as f:
        json.dump(asdict(log_data), f, indent=4)

    with open(output_dir / "rewards.json", "w") as f:
        json.dump([asdict(record) for record in reward_records], f, indent=4)


class BaseRunner(ABC):
    """Base class for algorithm runners."""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.reward_records: list[RewardRecord] = []
        self.log_data = LogData()

    @abstractmethod
    def setup_environment(self, env: Env) -> Env:
        """Setup the environment for the specific algorithm."""
        pass

    @abstractmethod
    def get_actions(self, env: Env) -> npt.NDArray[np.float64]:
        """Get actions for the current environment state."""
        pass

    @abstractmethod
    def get_output_dir(self) -> str:
        """Get the suffix for the output directory."""
        pass

    def run(self, data_path: str, output_dir: str, total_steps: int, seed: int, log_level: str) -> None:
        """Run the algorithm on the given data."""
        logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
        seed_everything(seed)

        # Load data and setup environment
        with open(data_path, "r") as f:
            raw_data = json.load(f)

        rl_log_data = LogData.from_json(raw_data)
        env = rl_log_data.get_env(min(rl_log_data.timestamps))
        env = self.setup_environment(env)

        # Run simulation
        pbar = tqdm(range(total_steps))
        for timestamp in pbar:
            start_time = time.perf_counter()
            actions = self.get_actions(env)
            time_taken = (time.perf_counter() - start_time) * 1000

            # Calculate reward and update records
            reward, reward_record = calc_reward(env, actions)
            reward_record.time_taken = time_taken
            reward_record.timestamp = timestamp
            self.reward_records.append(reward_record)

            # Update progress bar
            avg_reward = sum(r.reward for r in self.reward_records) / len(self.reward_records)
            pbar.set_postfix(reward=f"{reward:.4f}", avg=f"{avg_reward:.4f}")

            # Log and step environment
            self.log_data.append_state(env, actions, timestamp)
            env = env_step(env, actions)

        # Calculate and display final statistics
        stats = calculate_stats(self.reward_records)

        console = Console()
        table = Table(title=f"{self.algorithm_name} Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Total Reward", f"{stats['total_reward']:.4f}")
        table.add_row("Average Reward", f"{stats['average_reward']:.4f} ± {stats['reward_std']:.4f}")
        table.add_row("Mean Time", f"{stats['time_mean']:.4f} ± {stats['time_std']:.4f} ms")

        console.print(table)

        # Save results
        experiment_dir = Path(output_dir) / Path(data_path).stem / self.get_output_dir()
        save_results(self.log_data, self.reward_records, experiment_dir)
        console.print(f"Results saved to: {experiment_dir}")
