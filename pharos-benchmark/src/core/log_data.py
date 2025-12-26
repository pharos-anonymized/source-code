from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt

from core.env import Agent, Building, Env, Human


@dataclass
class HumanLog:
    hid: str

    position: list[float]
    velocity: list[float]

    ts: int

    @staticmethod
    def from_human(human: Human, timestamp: int) -> "HumanLog":
        return HumanLog(
            hid=human.id,
            position=human.position.tolist(),
            velocity=human.velocity.tolist(),
            ts=timestamp,
        )

    def to_human(self) -> Human:
        return Human(
            id=self.hid,
            position=np.array(self.position),
            velocity=np.array(self.velocity),
        )


@dataclass
class AgentLog:
    uid: str

    position: list[float]
    velocity: list[float]
    target_pos: list[float]

    include_area: list[float]
    action: list[float]

    ts: int

    @staticmethod
    def get_include_area(position: npt.NDArray[np.float64], action: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        cur_bbox_min = position - 0.5
        cur_bbox_max = position + 0.5

        next_bbox_min = (position + action) - 0.5
        next_bbox_max = (position + action) + 0.5

        bbox_min = np.minimum(cur_bbox_min, next_bbox_min)
        bbox_max = np.maximum(cur_bbox_max, next_bbox_max)

        return np.concatenate([bbox_min, bbox_max])

    @staticmethod
    def from_agent(agent: Agent, action: npt.NDArray[np.float64], timestamp: int) -> "AgentLog":
        return AgentLog(
            uid=agent.id,
            position=agent.position.tolist(),
            velocity=(action * 10).tolist(),
            target_pos=agent.target_pos.tolist(),
            include_area=AgentLog.get_include_area(agent.position, action).tolist(),
            action=action.tolist(),
            ts=timestamp,
        )

    def to_agent(self) -> Agent:
        return Agent(
            id=self.uid,
            position=np.array(self.position),
            velocity=np.array(self.velocity),
            target_pos=np.array(self.target_pos),
        )


@dataclass
class BuildingLog:
    id: str
    bbox: list[float]

    @staticmethod
    def from_building(building: Building) -> "BuildingLog":
        return BuildingLog(id=building.id, bbox=building.bbox.tolist())

    def to_building(self) -> Building:
        return Building(id=self.id, bbox=np.array(self.bbox))


@dataclass
class LogData:
    humans: list[HumanLog] = field(default_factory=list)
    devices: list[AgentLog] = field(default_factory=list)
    buildings: list[BuildingLog] = field(default_factory=list)

    @cached_property
    def timestamps(self):
        return sorted(set([agent.ts for agent in self.devices]))

    @staticmethod
    def from_json(raw_data: dict) -> "LogData":
        return LogData(
            humans=[HumanLog(**human) for human in raw_data["humans"]],
            devices=[AgentLog(**device) for device in raw_data["devices"]],
            buildings=[BuildingLog(**building) for building in raw_data["buildings"]],
        )

    def get_state(self, timestamp: int) -> tuple[list[HumanLog], list[AgentLog], list[BuildingLog]]:
        return (
            [human_log for human_log in self.humans if human_log.ts == timestamp],
            [agent_log for agent_log in self.devices if agent_log.ts == timestamp],
            [building_log for building_log in self.buildings],
        )

    def get_env(self, timestamp: int) -> Env:
        humans, agents, buildings = self.get_state(timestamp)
        return Env(
            humans=[human_log.to_human() for human_log in humans],
            agents=[agent_log.to_agent() for agent_log in agents],
            buildings=[building_log.to_building() for building_log in buildings],
        )

    def append_state(self, state: Env, actions: npt.NDArray[np.float64], timestamp: int):
        self.humans.extend([HumanLog.from_human(human, timestamp) for human in state.humans])
        self.devices.extend(
            [AgentLog.from_agent(agent, action, timestamp) for agent, action in zip(state.agents, actions)]
        )
        self.buildings.extend([BuildingLog.from_building(building) for building in state.buildings])
