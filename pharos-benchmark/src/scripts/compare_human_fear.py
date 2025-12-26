import statistics
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.env import RewardRecord
from core.utils import load_reward_data

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def calculate_fear_stats(reward_data: list[RewardRecord]) -> dict[str, float]:
    """Calculate fear and time statistics from reward data."""

    fear_penalties = [record.fear_penalty for record in reward_data]
    time_taken_values = [record.time_taken for record in reward_data if record.time_taken is not None]

    assert len(fear_penalties) == len(time_taken_values)
    assert len(fear_penalties) > 1

    total_fear = sum(fear_penalties)
    steps = len(fear_penalties)
    fear_per_step = total_fear / steps

    return {
        "fear_per_step": fear_per_step,
        "total_fear": total_fear,
        "mean_time": statistics.mean(time_taken_values),
        "std_time": statistics.stdev(time_taken_values),
        "steps": steps,
    }


@app.command()
def main(experiment_path: str = typer.Argument(..., help="Experiment directory path")):
    """Analyze fear per step, total fear and time taken for each algorithm in the experiment."""

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
        results[algo_dir.name] = calculate_fear_stats(reward_data)

    if not results:
        console.print("[red]No valid data found[/red]")
        raise typer.Exit(1)

    # Create results table
    table = Table(title=f"Fear and Time Analysis - {experiment_name}")
    columns = ["Algorithm", "Fear per Step", "Total Fear", "Time", "Steps"]
    styles = ["cyan", "magenta", "red", "yellow", "green"]

    for col, style in zip(columns, styles):
        table.add_column(col, style=style, justify="right" if col != "Algorithm" else "left")

    # Add sorted rows (by algorithm name)
    for algo, stats in sorted(results.items()):
        row = [
            algo,
            f"{stats['fear_per_step']:.4f}",
            f"{stats['total_fear']:.4f}",
            f"{stats['mean_time']:.4f} ± {stats['std_time']:.4f}",
            str(stats["steps"]),
        ]
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    app()
