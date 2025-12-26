import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort

from core.env import Agent, Env, calc_fear, dis_to_cube
from core.log_data import LogData

# Constants from the original script
ACTION_DIM = 7
OBS_DIM = 30


def load_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model and return the loaded model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")

    return ort.InferenceSession(model_path)


def predict(model: ort.InferenceSession, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference using ONNX model."""
    batch_size, obs_dim = obs.shape
    assert obs_dim == OBS_DIM, f"Expected obs dimension {OBS_DIM}, got {obs_dim}"

    # Convert numpy arrays to torch tensors
    obs_tensor = obs.astype(np.float32)
    available_actions_tensor = np.ones((batch_size, ACTION_DIM), dtype=np.float32)

    # Run inference
    actions, log_probs, probs = model.run(None, {"obs": obs_tensor, "available_actions": available_actions_tensor})

    # Convert back to numpy arrays
    return actions, log_probs, probs  # type: ignore


def normalize(arr: np.ndarray) -> np.ndarray:
    if np.linalg.norm(arr) == 0:
        return np.zeros_like(arr)
    return arr / np.linalg.norm(arr)


def potential_to_cube(pos: np.ndarray, cube_param: np.ndarray, K: float = 1.0) -> float:
    return K / (1 + dis_to_cube(pos, cube_param))


def get_obs(env: Env, agent: Agent, agent_id: int) -> np.ndarray:
    K = env.nearby_num
    min_bound = np.array(env.world_min_bound)
    max_bound = np.array(env.world_max_bound)

    # Other agent positions and velocities
    other_positions = np.array([a.position for a in env.agents if a != agent])
    other_velocities = np.array([a.velocity for a in env.agents if a != agent])

    # Self state
    obs = [agent_id]
    obs.extend(agent.position / (max_bound - min_bound))
    obs.append(np.sum(np.abs(agent.position - agent.target_pos)))
    obs.extend(normalize((agent.target_pos - agent.position) / (max_bound - min_bound)))

    # Find K nearest neighbors
    agent_dis = np.linalg.norm(agent.position - other_positions, axis=1)
    nearest_indices = np.argsort(agent_dis)[:K]

    # Neighbor features in 6 directions
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    neighbor_features = np.zeros(6, dtype=np.float32)

    for i, direction in enumerate(directions):
        for agent_pos, agent_vel in zip(other_positions[nearest_indices], other_velocities[nearest_indices]):
            neighbor_pos = agent.position + direction
            if np.array_equal(agent_pos, neighbor_pos):
                neighbor_features[i] += 1.0
            elif np.linalg.norm(agent_pos - neighbor_pos) == 1:
                rel_velocity = np.dot(agent_vel - agent.velocity, direction)
                neighbor_features[i] += 0.8 if rel_velocity > 0 else (0.5 if rel_velocity == 0 else 0.2)

    obs.extend(neighbor_features)

    # Fear values for current position and 6 directions
    next_step_fears = [calc_fear(env, agent.position, agent.velocity * env.agent_max_speed)]
    for direction in directions:
        fear = calc_fear(env, agent.position + direction, direction * env.agent_max_speed)
        next_step_fears.append(fear)

    next_step_fears = np.array(next_step_fears)
    next_step_fears -= np.mean(next_step_fears)
    obs.extend(next_step_fears)

    # Agent repulsion force
    agent_force_feat = np.zeros(3, dtype=np.float32)
    for neighbor_pos in other_positions[nearest_indices]:
        dist = np.linalg.norm(agent.position - neighbor_pos)
        agent_force_feat += (agent.position - neighbor_pos) / dist**3 if dist > 1e-6 else 0

    obs.extend(agent_force_feat)

    # Building potential gradients
    base_potential = sum(potential_to_cube(agent.position, b.bbox) for b in env.buildings)
    building_potentials = np.array(
        [sum(potential_to_cube(agent.position + d, b.bbox) for b in env.buildings) for d in directions]
    )
    building_potentials[1:] -= base_potential
    obs.extend(building_potentials)

    return np.array(obs, dtype=np.float32)


# TEST CODE
if __name__ == "__main__":
    data_path = "data/agents10_humans10_steps200_seed0.json"

    with open(data_path, "r") as f:
        raw_data = json.load(f)

    rl_log_data = LogData.from_json(raw_data)
    env = rl_log_data.get_env(min(rl_log_data.timestamps))
    env.action_discrete = True

    model = load_model("data/model.onnx")

    num_runs = 100

    # Time observation generation
    obs_times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        obs = np.array([get_obs(env, agent, id) for id, agent in enumerate(env.agents)])
        end_time = time.perf_counter()
        obs_times.append((end_time - start_time) * 1000)

    avg_obs_time = np.mean(obs_times)
    std_obs_time = np.std(obs_times)
    print(f"Observation generation took: {avg_obs_time:.3f} ± {std_obs_time:.3f} ms (avg ± std over {num_runs} runs)")

    # Time ONNX prediction
    pred_times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        actions, log_probs, probs = predict(model, obs)
        end_time = time.perf_counter()
        pred_times.append((end_time - start_time) * 1000)

    avg_pred_time = np.mean(pred_times)
    std_pred_time = np.std(pred_times)
    print(f"ONNX prediction took: {avg_pred_time:.3f} ± {std_pred_time:.3f} ms (avg ± std over {num_runs} runs)")

    print(f"Observation shape: {obs.shape}")
    print(f"Actions shape: {actions.shape}, Log probs shape: {log_probs.shape}, Probs shape: {probs.shape}")
    print(f"Actions: {actions.T}")
    print(f"Log probs: {log_probs.T}")
