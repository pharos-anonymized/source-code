import numpy as np
import numpy.typing as npt

from algorithms.rl_model import get_obs, load_model, predict
from core.env import Env

from .base_runner import BaseRunner

ACTION_MAPPING = {
    0: [0, 0, 0],  # STAY
    1: [0, 0, 1],  # UP
    2: [0, 0, -1],  # DOWN
    3: [-1, 0, 0],  # LEFT
    4: [1, 0, 0],  # RIGHT
    5: [0, 1, 0],  # FORWARD
    6: [0, -1, 0],  # BACKWARD
}


class PharosRunner(BaseRunner):
    """Pharos RL algorithm runner."""

    def __init__(self, model_path: str, seed: int):
        super().__init__("Pharos")
        self.model_path = model_path
        self.seed = seed
        self.model = None

    def setup_environment(self, env: Env) -> Env:
        """Setup the environment for Pharos algorithm."""
        env.action_discrete = True
        self.model = load_model(self.model_path)
        return env

    def get_actions(self, env: Env) -> npt.NDArray[np.float64]:
        """Get actions using Pharos RL model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup_environment first.")

        obs = np.array([get_obs(env, agent, id) for id, agent in enumerate(env.agents)])
        actions, log_probs, probs = predict(self.model, obs)

        actions = np.array([ACTION_MAPPING[a] for a in actions.flatten().tolist()])

        for i, agent in enumerate(env.agents):
            next_pos = agent.position + actions[i]
            if next_pos[1] < 0.0:
                actions[i] = [0, 0, 0]

        return actions.astype(np.float64)

    def get_output_dir(self) -> str:
        """Get the suffix for the output directory."""
        return f"pharos_seed{self.seed}"
