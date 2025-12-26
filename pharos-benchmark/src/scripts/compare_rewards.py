import statistics
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.env import RewardRecord
from core.utils import load_reward_data

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def calculate_reward_stats(reward_data: list[RewardRecord]) -> dict[str, float]:
    """Calculate reward and time statistics from reward data."""

    rewards = [record.reward for record in reward_data]
    time_taken_values = [record.time_taken for record in reward_data if record.time_taken is not None]

    assert len(rewards) == len(time_taken_values)
    assert len(rewards) > 1

    return {
        "mean_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards),
        "mean_time": statistics.mean(time_taken_values),
        "std_time": statistics.stdev(time_taken_values),
        "steps": len(rewards),
    }


@app.command()
def main(experiment_path: str = typer.Argument(..., help="Experiment directory path")):
    """Analyze reward and time statistics for each algorithm in the experiment."""

    # Find algorithm directories with rewards.json
    algo_dirs = [d for d in Path(experiment_path).iterdir() if d.is_dir() and (d / "rewards.json").exists()]
    experiment_name = Path(experiment_path).name

    if not algo_dirs:
        console.print(f"[red]No algorithm directories found in {experiment_path}[/red]")
        raise typer.Exit(1)

    # Analyze each algorithm
    results = {}
    for algo_dir in sorted(algo_dirs):
        reward_data = load_reward_data(algo_dir / "rewards.json")
        results[algo_dir.name] = calculate_reward_stats(reward_data)

    if not results:
        console.print("[red]No valid data found[/red]")
        raise typer.Exit(1)

    # Create results table
    table = Table(title=f"Reward and Time Analysis - {experiment_name}")
    columns = ["Algorithm", "Reward", "Time", "Steps"]
    styles = ["cyan", "magenta", "red", "yellow"]

    for col, style in zip(columns, styles):
        table.add_column(col, style=style, justify="right" if col != "Algorithm" else "left")

    # Add sorted rows (by algorithm name)
    for algo, stats in sorted(results.items()):
        row = [
            algo,
            f"{stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}",
            f"{stats['mean_time']:.4f} ± {stats['std_time']:.4f}",
            str(stats["steps"]),
        ]
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    app()
