import json
import math
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from core.log_data import LogData

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def calculate_entropy(log_data: LogData):
    # Count agent appearances at each integer coordinate
    coord_counts = defaultdict(int)
    total_steps = 0

    for timestamp in log_data.timestamps:
        _, agents, _ = log_data.get_state(timestamp)
        total_steps += 1

        for agent in agents:
            # Convert to integer coordinates (x, y, z)
            int_x = int(round(agent.position[0]))
            int_y = int(round(agent.position[1]))
            int_z = int(round(agent.position[2]))
            coord_counts[(int_x, int_y, int_z)] += 1

    # Calculate entropy: -sum(p * log(p))
    entropy = 0.0
    for count in coord_counts.values():
        if count > 0:
            p = count / total_steps
            entropy -= p * math.log(p)

    return entropy


@app.command()
def main(experiment_dir: str = typer.Argument(..., help="Experiment directory path")):
    experiment_path = Path(experiment_dir)
    algo_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and (d / "vis_data.json").exists()]

    if not algo_dirs:
        console.print(f"[red]No algorithm directories found in {experiment_path.name}[/red]")
        raise typer.Exit(1)

    results = {}
    for algo_dir in algo_dirs:
        log_data = LogData.from_json(json.loads((algo_dir / "vis_data.json").read_text()))
        results[algo_dir.name] = calculate_entropy(log_data)

    table = Table(title=f"Entropy Comparison: {experiment_path.name}")
    table.add_column("Algorithm", style="cyan", justify="left")
    table.add_column("Entropy", style="magenta", justify="right")

    for algo, entropy in sorted(results.items()):
        table.add_row(algo, f"{entropy:.4f}")

    console.print(table)


if __name__ == "__main__":
    app()
