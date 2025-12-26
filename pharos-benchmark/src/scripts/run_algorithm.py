import typer
from typing_extensions import Annotated

from runner import AStarRunner, PharosRunner, SolverRunner

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def astar(
    data_path: Annotated[str, typer.Argument(help="Path to the data file")],
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output",
    total_steps: Annotated[int, typer.Option("--steps", help="Total steps")] = 200,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    log_level: Annotated[str, typer.Option("--log-level", help="Log level")] = "INFO",
):
    runner = AStarRunner(seed)
    typer.echo("🚀 Running A* algorithm...")
    runner.run(data_path, output_dir, total_steps, seed, log_level)
    typer.echo("✅ A* algorithm completed successfully!")


@app.command()
def pharos(
    data_path: Annotated[str, typer.Argument(help="Path to the data file")],
    model_path: Annotated[str, typer.Option("--model", help="Path to the model file")] = "data/model.onnx",
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output",
    total_steps: Annotated[int, typer.Option("--steps", help="Total steps")] = 200,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    log_level: Annotated[str, typer.Option("--log-level", help="Log level")] = "INFO",
):
    runner = PharosRunner(model_path, seed)
    typer.echo("🚀 Running PHAROS algorithm...")
    runner.run(data_path, output_dir, total_steps, seed, log_level)
    typer.echo("✅ PHAROS algorithm completed successfully!")


@app.command()
def solver(
    data_path: Annotated[str, typer.Argument(help="Path to the data file")],
    solver_name: Annotated[str, typer.Option("--solver", help="Solver to use")] = "ipopt",
    action_discrete: Annotated[bool, typer.Option("--discrete", help="Use discrete action space")] = False,
    output_dir: Annotated[str, typer.Option("--output", "-o", help="Output directory")] = "output",
    total_steps: Annotated[int, typer.Option("--steps", help="Total steps")] = 200,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    log_level: Annotated[str, typer.Option("--log-level", help="Log level")] = "INFO",
):
    runner = SolverRunner(solver_name, action_discrete, seed)
    typer.echo("🚀 Running SOLVER algorithm...")
    runner.run(data_path, output_dir, total_steps, seed, log_level)
    typer.echo("✅ SOLVER algorithm completed successfully!")


if __name__ == "__main__":
    app()
