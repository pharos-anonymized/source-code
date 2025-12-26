import {
  BuildingData,
  DeviceData,
  HumanData,
  RawBuildingData,
  RawDeviceData,
  RawHumanData,
} from "@/types";
import { Vector3 } from "three";

const toCuboid = (bbox: [number, number, number, number, number, number]) => {
  const minPoint = new Vector3(bbox[0], bbox[1], bbox[2]);
  const maxPoint = new Vector3(bbox[3], bbox[4], bbox[5]);
  return {
    position: minPoint.clone().add(maxPoint).multiplyScalar(0.5),
    size: maxPoint.clone().sub(minPoint),
    minPoint,
    maxPoint,
  };
};

export const toDeviceData = (rawData: RawDeviceData[]): DeviceData[] => {
  return rawData
    .map((data) => ({
      uid: data.uid,
      timestamp: data.ts,
      position: new Vector3(...data.position),
      velocity: new Vector3(...data.velocity),
      safeSpace: toCuboid(data.include_area),
      targetPos: new Vector3(...data.target_pos),
    }))
    .sort((a, b) => a.timestamp - b.timestamp);
};

export const toHumanData = (rawData: RawHumanData[]): HumanData[] => {
  return rawData.map((data) => ({
    hid: data.hid,
    timestamp: data.ts,
    position: new Vector3(...data.position),
    velocity: new Vector3(...data.velocity),
  }));
};

export const toBuildingData = (rawData: RawBuildingData[]): BuildingData[] => {
  return rawData.map((data) => ({
    id: data.id,
    bbox: toCuboid(data.bbox),
  }));
};
