import logging

import numpy as np
import numpy.typing as npt

from algorithms.pyomo_solver import PyomoSolver
from core.env import Env

from .base_runner import BaseRunner


class SolverRunner(BaseRunner):
    """Optimization solver algorithm runner."""

    def __init__(self, solver_name: str, action_discrete: bool, seed: int):
        super().__init__(solver_name.upper())
        self.solver_name = solver_name
        self.action_discrete = action_discrete
        self.seed = seed

    def setup_environment(self, env: Env) -> Env:
        """Setup the environment for solver algorithm."""
        env.action_discrete = self.action_discrete
        return env

    def get_actions(self, env: Env) -> npt.NDArray[np.float64]:
        """Get actions using optimization solver."""
        solver = PyomoSolver(solver_name=self.solver_name)
        _, actions, success, results = solver.solve_single_step(env)

        action_norms = np.linalg.norm(actions, axis=1)
        actions[action_norms < 1e-2] = 0.0

        # Sometimes the solver fails to solve the problem, but it can continue to run
        if not success:
            logging.error("Failed to solve optimization problem")
            logging.error(results)

        return actions

    def get_output_dir(self) -> str:
        """Get the suffix for the output directory."""
        action_type = "discrete" if self.action_discrete else "continuous"
        return f"{self.solver_name}_{action_type}_seed{self.seed}"
