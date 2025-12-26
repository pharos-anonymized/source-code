from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Human:
    id: str

    position: np.ndarray
    velocity: np.ndarray


@dataclass
class Agent:
    id: str

    position: np.ndarray
    velocity: np.ndarray
    target_pos: np.ndarray


@dataclass
class Building:
    id: str
    bbox: np.ndarray

    @property
    def bbox_min(self) -> np.ndarray:
        return self.bbox[:3]

    @property
    def bbox_max(self) -> np.ndarray:
        return self.bbox[3:]


@dataclass
class Env:
    # States
    humans: list[Human]
    agents: list[Agent]
    buildings: list[Building]

    # Range of world
    world_min_bound: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    world_max_bound: np.ndarray = field(default_factory=lambda: np.array([30.0, 10.0, 30.0]))

    # Range of agent start point and target point
    target_min_bound: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    target_max_bound: np.ndarray = field(default_factory=lambda: np.array([30, 10, 30]))

    # Reward config
    closer_factor: float = 10.0
    reach_factor: float = 30.0
    scare_factor: float = 25.0
    collision_factor: float = 1000.0
    cutoff_scare_distance: float = 5.0

    # Action config
    action_discrete: bool = False

    # Agent config
    agent_max_speed: float = 10.0

    # Obs Config
    nearby_num: int = 9

    # Human config
    human_speed_mean: float = 1.34
    human_speed_std: float = 0.37
    human_velocity_update_interval: float = 100.0  # Average number of steps before velocity update


def dis_to_cube(point: np.ndarray, bbox: np.ndarray) -> float:
    dx = np.maximum(0, np.maximum(bbox[0] - point[0], point[0] - bbox[3]))
    dy = np.maximum(0, np.maximum(bbox[1] - point[1], point[1] - bbox[4]))
    dz = np.maximum(0, np.maximum(bbox[2] - point[2], point[2] - bbox[5]))
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def generate_new_target(env: Env) -> np.ndarray:
    MIN_DISTANCE = 1
    MAX_ATTEMPTS = 10000

    for _ in range(0, MAX_ATTEMPTS):
        target = np.random.randint(low=env.target_min_bound, high=env.target_max_bound)
        if all(dis_to_cube(target, building.bbox) >= MIN_DISTANCE for building in env.buildings):
            return target

    raise RuntimeError(f"Failed to generate valid target after {MAX_ATTEMPTS} attempts")


def generate_human_velocity(env: Env) -> np.ndarray:
    direction = np.random.uniform(0, 2 * np.pi)
    speed = np.random.normal(env.human_speed_mean, env.human_speed_std)
    cos_dir, sin_dir = np.cos(direction), np.sin(direction)
    return speed * np.array([cos_dir, 0.0, sin_dir])


def env_step(env: Env, actions: np.ndarray, delta_t: float = 0.1) -> Env:
    env_next = deepcopy(env)

    for human in env_next.humans:
        new_position = human.position + human.velocity * delta_t
        human.position = new_position

        if any(human.position > env.world_max_bound) or any(human.position < env.world_min_bound):
            human.position = np.clip(human.position, env.world_min_bound, env.world_max_bound)
            human.velocity = -human.velocity

        if np.random.random() < (1.0 / env.human_velocity_update_interval):
            human.velocity = generate_human_velocity(env)

    for agent, action in zip(env_next.agents, actions):
        agent.position = agent.position + action
        agent.velocity = action * env.agent_max_speed

        if np.linalg.norm(agent.position - agent.target_pos) < 0.5:
            agent.target_pos = generate_new_target(env)

    return env_next


@dataclass
class RewardRecord:
    closer_reward: float
    reach_reward: float
    collision_penalty: float
    fear_penalty: float

    reward: float

    a2h_collision: int
    a2b_collision: int
    a2a_collision: int

    timestamp: int | None = None
    time_taken: float | None = None

    def todict(self) -> dict:
        return {**asdict(self), "reward": self.reward}


def calc_reward(env: Env, actions: np.ndarray) -> tuple[float, RewardRecord]:
    reward = 0.0

    curr_pos = np.array([agent.position for agent in env.agents])
    next_pos = curr_pos + actions
    next_vel = actions * env.agent_max_speed

    # 1. Target closer reward
    target_pos = np.array([agent.target_pos for agent in env.agents])
    curr_dis_to_target = np.linalg.norm(curr_pos - target_pos, axis=1)
    next_dis_to_target = np.linalg.norm(next_pos - target_pos, axis=1)

    closer_reward = np.sum((curr_dis_to_target - next_dis_to_target) * env.closer_factor)

    # 2. Reach reward
    reach_threshold = 1e-2
    distances_to_target = np.linalg.norm(next_pos - target_pos, axis=1)
    reach_reward = np.sum(distances_to_target < reach_threshold) * env.reach_factor

    # 3. Collision penalty
    collision_threshold = 0.5  # Distance threshold for collision
    a2h_collision, a2b_collision, a2a_collision = 0, 0, 0

    # Agent-Human collisions
    for i, agent in enumerate(env.agents):
        for human in env.humans:
            distance = np.linalg.norm(next_pos[i] - human.position)
            if distance < collision_threshold:
                a2h_collision += 1

    # Agent-Building collisions
    for i, agent in enumerate(env.agents):
        for building in env.buildings:
            agent_pos = next_pos[i]
            agent_min, agent_max = agent_pos - 0.5, agent_pos + 0.5
            building_min, building_max = building.bbox_min, building.bbox_max
            if np.all(np.logical_and(agent_min <= building_max, agent_max >= building_min)):
                a2b_collision += 1

    # Agent-Agent collisions
    for i, agent1 in enumerate(env.agents):
        for j, agent2 in enumerate(env.agents[i + 1 :], i + 1):
            distance = np.linalg.norm(next_pos[i] - next_pos[j])
            if distance < collision_threshold:
                a2a_collision += 1

    collision_penalty = (a2h_collision + a2b_collision + a2a_collision) * env.collision_factor

    # 4. Fear penalty
    fearness = sum(calc_fear(env, next_pos[i], next_vel[i]) for i in range(len(env.agents)))
    fear_penalty = fearness * env.scare_factor

    # Calculate total reward
    reward = closer_reward + reach_reward - collision_penalty - fear_penalty

    record = RewardRecord(
        closer_reward=round(float(closer_reward), 4),
        reach_reward=round(float(reach_reward), 4),
        collision_penalty=round(float(collision_penalty), 4),
        fear_penalty=round(float(fear_penalty), 4),
        reward=round(float(reward), 4),
        a2h_collision=a2h_collision,
        a2b_collision=a2b_collision,
        a2a_collision=a2a_collision,
    )

    return float(reward), record


def calc_fear(env: Env, agent_pos: np.ndarray, agent_vel: np.ndarray) -> float:
    if np.linalg.norm(agent_vel) < 1e-6 or not env.humans:
        return 0.0

    human_positions = np.array([h.position for h in env.humans])
    human_positions[:, 1] = 1.7
    human_velocities = np.array([h.velocity for h in env.humans])

    # Calculate cosine similarity of velocities
    distances: np.ndarray = np.linalg.norm(human_positions - agent_pos, axis=1)
    distances = np.maximum(distances, 1e-6)
    agent_velocity_norm = np.maximum(np.linalg.norm(agent_vel), 1e-6)
    human_velocity_norms = np.maximum(np.linalg.norm(human_velocities, axis=1), 1e-6)
    cosines_v = np.dot(human_velocities, agent_vel) / (human_velocity_norms * agent_velocity_norm)

    # Calculate cosine of relative position and velocity
    rel_pos = agent_pos[np.newaxis, :] - human_positions
    rel_vel = agent_vel[np.newaxis, :] - human_velocities
    rel_pos_norm = np.maximum(np.linalg.norm(rel_pos, axis=1), 1e-6)
    rel_vel_norm = np.maximum(np.linalg.norm(rel_vel, axis=1), 1e-6)
    cosines_r = (rel_vel * rel_pos).sum(axis=1) / (rel_vel_norm * rel_pos_norm)

    # Calculate fearness value for each human
    # fearness = 1/dij * (1 - cos(vi, vj)) * max(0, cos(vji, rji))
    fearness = (1 / distances) * (1 - cosines_v) * np.maximum(0, cosines_r)

    # Apply distance cutoff - no fear beyond cutoff distance
    fearness = np.where(distances > env.cutoff_scare_distance, 0, fearness)

    return np.sum(fearness)
