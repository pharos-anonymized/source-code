import heapq

import numpy as np

from core.env import Agent, Env, dis_to_cube


class GridNode:
    """A node in the 3D grid for A* pathfinding."""

    def __init__(self, x: int, y: int, z: int):
        self.x, self.y, self.z = x, y, z

    def __eq__(self, other: "GridNode") -> bool:
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __lt__(self, other: "GridNode") -> bool:
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"GridNode({self.x}, {self.y}, {self.z})"

    def to_position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_position(cls, position: np.ndarray) -> "GridNode":
        grid_pos = np.round(position).astype(int)
        return cls(*grid_pos)


class AStarPathfinder:
    """A* pathfinder for 3D navigation with obstacle avoidance."""

    def __init__(self, env: Env):
        super().__init__()
        self.env = env
        self.collision_threshold = 0.5

        self._occupied_positions: list[np.ndarray] = []

        self._world_building_valid: dict[tuple, bool] = {}
        self._agent_agent_valid: dict[tuple, bool] = {}
        self._agent_human_valid: dict[tuple, bool] = {}

    def is_valid_position(self, position: np.ndarray) -> bool:
        """Check if position is within bounds and collision-free."""

        pos_key = tuple(np.round(position, 2).tolist())

        # Check world bounds and building collisions
        if pos_key not in self._world_building_valid:
            if not (np.all(position >= self.env.world_min_bound) and np.all(position <= self.env.world_max_bound)):
                self._world_building_valid[pos_key] = False
                return False

            building_bboxes = np.array([building.bbox for building in self.env.buildings])
            if np.any(dis_to_cube(position, building_bboxes.T) < self.collision_threshold):
                self._world_building_valid[pos_key] = False
                return False

        if pos_key in self._world_building_valid:
            return False

        # Check agent-human collisions
        if pos_key not in self._agent_human_valid:
            human_positions = np.array([human.position for human in self.env.humans])
            distances = np.linalg.norm(human_positions - position, axis=1)
            if np.any(distances < self.collision_threshold):
                self._agent_human_valid[pos_key] = False
                return False

        if pos_key in self._agent_human_valid:
            return False

        # Check agent-agent collisions
        if pos_key not in self._agent_agent_valid:
            occupied_positions = np.array(self._occupied_positions + [agent.position for agent in self.env.agents])
            distances = np.linalg.norm(occupied_positions - position, axis=1)
            self._agent_agent_valid[pos_key] = not np.any(distances < self.collision_threshold)

        return self._agent_agent_valid[pos_key]

    def neighbors(self, node: GridNode):
        """Get valid neighboring nodes in 6 directions only."""
        directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        neighbors: list[GridNode] = []
        for dx, dy, dz in directions:
            new_node = GridNode(node.x + dx, node.y + dy, node.z + dz)
            if self.is_valid_position(new_node.to_position()):
                neighbors.append(new_node)

        return neighbors

    def distance_between(self, n1: GridNode, n2: GridNode) -> float:
        """Manhattan distance between adjacent nodes."""
        return float(abs(n1.x - n2.x) + abs(n1.y - n2.y) + abs(n1.z - n2.z))

    def heuristic_cost_estimate(self, current: GridNode, goal: GridNode) -> float:
        """Manhattan distance heuristic."""
        dx, dy, dz = abs(current.x - goal.x), abs(current.y - goal.y), abs(current.z - goal.z)
        return float(dx + dy + dz)

    def find_path(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        """Find path from start to goal using A*."""
        start_node = GridNode.from_position(start)
        goal_node = GridNode.from_position(goal)

        path_nodes = self.astar(start_node, goal_node)
        return [node.to_position() for node in path_nodes] if path_nodes else None

    def get_action(self, agent: Agent) -> np.ndarray:
        """Get action for agent considering other agent positions."""
        path = self.find_path(agent.position, agent.target_pos)

        if not path or len(path) < 2:
            return np.zeros(3)

        return path[1] - agent.position

    def get_actions_batch(self, agents: list[Agent]) -> list[np.ndarray]:
        """Get actions for all agents sequentially to prevent collisions."""
        actions = []
        self._occupied_positions = []
        self._agent_human_valid.clear()

        for agent in agents:
            self._agent_agent_valid.clear()
            action = self.get_action(agent)
            actions.append(action)

            # Add the next position this agent will occupy
            next_position = agent.position + action
            self._occupied_positions.append(next_position)

        return actions

    def reconstruct_path(self, came_from: dict[GridNode, GridNode], start: GridNode, node: GridNode) -> list[GridNode]:
        path = []
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append(start)
        return list(reversed(path))

    def astar(self, start: GridNode, goal: GridNode, max_nodes_explored: int = 50) -> list[GridNode] | None:
        """A* pathfinding algorithm implementation."""

        # Initialize data structures for A* search
        open_set: list[tuple[float, GridNode]] = [(0, start)]
        came_from: dict[GridNode, GridNode] = {}
        g_score: dict[GridNode, float] = {start: 0}
        closed_set: set[GridNode] = set()

        while open_set and len(closed_set) < max_nodes_explored:
            # Get node with lowest f-score
            current_f, current = heapq.heappop(open_set)

            # Check if we reached the goal
            if current == goal:
                return self.reconstruct_path(came_from, start, current)

            closed_set.add(current)

            # Explore all neighbors
            for neighbor in self.neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + self.distance_between(current, neighbor)

                # Update path if we found a better route
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic_cost_estimate(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        # If no path found, return path to the closest point we reached
        if closed_set:
            closest_node = min(closed_set, key=lambda node: self.heuristic_cost_estimate(node, goal))
            return self.reconstruct_path(came_from, start, closest_node)

        return None
