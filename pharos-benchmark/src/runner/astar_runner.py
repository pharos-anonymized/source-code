import numpy as np
import numpy.typing as npt

from algorithms.astar import AStarPathfinder
from core.env import Env

from .base_runner import BaseRunner


class AStarRunner(BaseRunner):
    """A* algorithm runner."""

    def __init__(self, seed: int):
        super().__init__("A*")
        self.seed = seed
        self.pathfinder: AStarPathfinder | None = None

    def setup_environment(self, env: Env) -> Env:
        """Setup the environment for A* algorithm."""
        self.pathfinder = AStarPathfinder(env)
        return env

    def get_actions(self, env: Env) -> npt.NDArray[np.float64]:
        """Get actions using A* pathfinding."""
        if self.pathfinder is None:
            raise RuntimeError("Pathfinder not initialized. Call setup_environment first.")

        self.pathfinder.env = env
        actions_list = self.pathfinder.get_actions_batch(env.agents)
        return np.array(actions_list, dtype=np.float64)

    def get_output_dir(self) -> str:
        """Get the suffix for the output directory."""
        return f"astar_seed{self.seed}"
