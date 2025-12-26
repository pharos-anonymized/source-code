import json
import random
from os import PathLike

import numpy as np
import typer
from rich.console import Console

from core.env import RewardRecord

console = Console()


def load_reward_data(file_path: PathLike | str) -> list[RewardRecord]:
    """Load reward data from a JSON file."""
    try:
        with open(file_path, "r") as f:
            data_raw = json.load(f)
            return [RewardRecord(**record) for record in data_raw]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[red]Error loading {file_path}: {e}[/red]")
        raise typer.Exit(1)


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
