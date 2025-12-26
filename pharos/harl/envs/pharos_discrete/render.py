# -*- coding: utf-8 -*-
import json
from typing import Dict, List, Optional
import numpy as np

"""
{
  "devices": [
    {
      "uid": "vehicle/10001",
      "position": [24.0, 48.0, 26.0],
      "velocity": [0.0, 0.0, 0.0],
      "ts": 1742393414649,
      "include_area": [22.0, 46.0, 24.0, 26.0, 50.0, 28.0],
    },
    {
      "uid": "vehicle/10002",
      "position": [31.0, 25.0, 50.0],
      "velocity": [0.0, 0.0, 0.0],
      "ts": 1742393414649,
      "include_area": [29.0, 23.0, 48.0, 33.0, 27.0, 52.0],
    },
    // ...
  ],
  "humans": [
    {
      "hid": "human/10001",
      "position": [24.0, 1.0, 26.0],
      "velocity": [1.0, 0.0, 0.0],
      "ts": 1742393414649,
    },
    // ...
  ]
}
数据文件中包含的设备数据结构如下：

uid 为设备ID
position 为设备位置
velocity 为设备速度
ts 为设备时间戳，自动播放时每 0.1 秒加 100
include_area 为安全空间，minX, minY, minZ, maxX, maxY, maxZ


"""


class AgentVis:
    def __init__(
        self,
        uid: str,
        position: np.ndarray[(3,), float],
        velocity: np.ndarray[(3,), float],
        ts: int,
        include_area: np.ndarray[(6,), float],
        target_pos: Optional[np.ndarray[(3,), float]] = None,
        prev_action: Optional[int] = None,
    ):
        self.uid = uid
        self.position = position.tolist()
        self.velocity = velocity.tolist()
        self.ts = ts
        self.include_area = include_area.tolist()
        self.target_pos = target_pos.tolist() if target_pos is not None else None
        self.prev_action = prev_action if prev_action is not None else None

    def __str__(self):
        return f"Device(uid={self.uid}, position={self.position}, velocity={self.velocity}, ts={self.ts}, \
            include_area={self.include_area}, target_pos={self.target_pos}), prev_action={self.prev_action})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_dict(data):
        return AgentVis(
            uid=data["uid"],
            position=data["position"].tolist(),
            velocity=data["velocity"].tolist(),
            ts=data["ts"],
            include_area=data["include_area"].tolist(),
            target_pos=data.get("target_pos", None),
            prev_action=data.get("prev_action", None),
        )

    def to_dict(self):
        return {
            "uid": self.uid,
            "position": self.position,
            "velocity": self.velocity,
            "ts": self.ts,
            "include_area": self.include_area,
            "target_pos": self.target_pos,
            "prev_action": self.prev_action,  # Previous action, if needed for rendering or debugging
        }

    def to_json(self):
        return json.dumps(self.to_dict())


class HumanVis:
    def __init__(
        self,
        hid: str,
        position: np.ndarray[(3,), float],
        velocity: np.ndarray[(3,), float],
        ts: int,
    ):
        self.hid = hid
        self.position = position.tolist()
        self.velocity = velocity.tolist()
        self.ts = ts

    def __str__(self):
        return f"Human(hid={self.hid}, position={self.position}, velocity={self.velocity}, ts={self.ts})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_dict(data):
        return HumanVis(
            hid=data["hid"],
            position=data["position"].tolist(),
            velocity=data["velocity"].tolist(),
            ts=data["ts"],
        )

    def to_dict(self):
        return {
            "hid": self.hid,
            "position": self.position,
            "velocity": self.velocity,
            "ts": self.ts,
        }

    def to_json(self):
        return json.dumps(self.to_dict())


def save_json(
    path: str,
    data: List[AgentVis],
    human_data: Optional[List[HumanVis]] = None,
    building_data: Optional[List[Dict]] = None,
    append: bool = False,
):
    # print(
    #     f"Saved {len(data)} steps devices,  {len(human_data) if human_data else 0} humans, "
    #     f"{len(building_data) if building_data else 0} buildings to {path}"
    # )
    # print(f"Max ts: {max(device.ts for device in data)}")
    # print(f"Different ts count: {len(set(device.ts for device in data))}")

    # 检查 data 是否为 AgentVis 类型的对象列表
    # print(f"data: {data}")
    # print(f"human_data: {human_data}")
    if not all(isinstance(device, AgentVis) for device in data):
        raise TypeError("`data` 参数必须是由 `AgentVis` 对象组成的列表")

    # 检查 human_data 是否为 HumanVis 类型的对象列表
    if human_data is not None and not all(
        isinstance(human, HumanVis) for human in human_data
    ):
        raise TypeError("`human_data` 参数必须是由 `HumanVis` 对象组成的列表")
    mode = "a" if append else "w"
    with open(path, mode) as f:
        all_data = {}
        all_data["devices"] = [device.to_dict() for device in data]
        if human_data is not None:
            all_data["humans"] = [human.to_dict() for human in human_data]
        else:
            all_data["humans"] = []
        if building_data is not None:
            all_data["buildings"] = building_data
        else:
            all_data["buildings"] = []
        json.dump(all_data, f, indent=4)
