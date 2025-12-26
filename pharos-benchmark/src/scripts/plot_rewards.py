from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from rich.console import Console

from core.env import RewardRecord
from core.utils import load_reward_data

app = typer.Typer()
console = Console()


def extract_algorithm_data(data: list[RewardRecord], window_size: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract and process data for a single algorithm."""
    timestamps = np.array([r.timestamp for r in data])
    timestamps = timestamps - np.min(timestamps)  # Normalize to start from 0
    rewards = np.array([r.reward for r in data])

    df = pd.DataFrame({"timestamp": timestamps, "reward": rewards})
    df["moving_mean"] = df["reward"].rolling(window=window_size, center=True, min_periods=1).mean()
    df["moving_std"] = df["reward"].rolling(window=window_size, center=True, min_periods=1).std()

    return timestamps, df


def plot_reward_comparison(
    pharos_data: list[RewardRecord],
    algo_data: list[RewardRecord],
    pharos_label: str,
    algo_label: str,
    window_size: int,
):
    """Plot reward comparison between pharos and any other algorithm data with moving variance."""

    # Extract and process data
    pharos_ts, pharos_df = extract_algorithm_data(pharos_data, window_size)
    algo_ts, algo_df = extract_algorithm_data(algo_data, window_size)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"pharos": "#2E86AB", "algo": "#A23B72", "pharos_fill": "#2E86AB", "algo_fill": "#A23B72"}

    # Plot pharos data with moving average
    ax.plot(pharos_ts, pharos_df["moving_mean"], label=pharos_label, color=colors["pharos"], linewidth=2, alpha=0.9)

    # Add pharos confidence interval
    pharos_upper = pharos_df["moving_mean"] + pharos_df["moving_std"]
    pharos_lower = pharos_df["moving_mean"] - pharos_df["moving_std"]
    ax.fill_between(
        pharos_ts, pharos_lower, pharos_upper, alpha=0.15, color=colors["pharos_fill"], label=f"{pharos_label} ±1σ"
    )

    # Plot algorithm data with moving average
    ax.plot(algo_ts, algo_df["moving_mean"], label=algo_label, color=colors["algo"], linewidth=2, alpha=0.9)

    # Add algorithm confidence interval
    algo_upper = algo_df["moving_mean"] + algo_df["moving_std"]
    algo_lower = algo_df["moving_mean"] - algo_df["moving_std"]
    ax.fill_between(algo_ts, algo_lower, algo_upper, alpha=0.15, color=colors["algo_fill"], label=f"{algo_label} ±1σ")

    # Beautify axes and title
    ax.set_xlim(min(pharos_ts), max(pharos_ts) + 100)
    ax.set_xlabel("Time (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reward", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Performance Comparison: {pharos_label} vs {algo_label}\n(Moving Average Window Size: {window_size})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Improve legend style
    legend = ax.legend(loc="best", frameon=True, fancybox=True, fontsize=10, framealpha=0.9)
    legend.get_frame().set_facecolor("white")

    # Improve grid style
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_facecolor("#fafafa")

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("gray")

    plt.tight_layout()


@app.command()
def main(
    pharos_data_path: str = typer.Argument(..., help="Path to Pharos reward data JSON file"),
    algo_data_path: str = typer.Argument(..., help="Path to comparison algorithm reward data JSON file"),
    pharos_label: str = typer.Option("Pharos", help="Label for Pharos in plot legend"),
    algo_label: str = typer.Option("Solver", help="Label for comparison algorithm in plot legend"),
    slice_start: int = typer.Option(0, help="Starting index for data slice"),
    slice_end: int = typer.Option(-1, help="Ending index for data slice (-1 for end)"),
    window_size: int = typer.Option(10, help="Window size for moving average"),
    output_path: Path = typer.Option(Path("output/figures/reward_comparison.png"), help="Output plot path"),
    dpi: int = typer.Option(600, help="Output image resolution (DPI)"),
    format: str = typer.Option("png", help="Output format (png, pdf, svg)"),
):
    pharos_data = load_reward_data(pharos_data_path)
    algo_data = load_reward_data(algo_data_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the plot
    slice_ = slice(slice_start, slice_end)
    plot_reward_comparison(pharos_data[slice_], algo_data[slice_], pharos_label, algo_label, window_size)

    # Save with specified format
    output_path = output_path.with_suffix(f".{format}")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", format=format)
    print(f"Plot saved to: {output_path.absolute()}")


if __name__ == "__main__":
    app()
