import { Vector3 } from "three";

export type Cuboid = {
  size: Vector3; // size along x, y, z axes
  position: Vector3; // center point of the cuboid
  minPoint: Vector3; // minimum (x, y, z) coordinates
  maxPoint: Vector3; // maximum (x, y, z) coordinates
};

export type DeviceData = {
  uid: string;
  timestamp: number;
  position: Vector3;
  velocity: Vector3;
  safeSpace: Cuboid;
  targetPos: Vector3;
};

export type HumanData = {
  hid: string;
  timestamp: number;
  position: Vector3;
  velocity: Vector3;
};

export type BuildingData = {
  id: string;
  bbox: Cuboid;
};

export type RawDeviceData = {
  uid: string;
  ts: number;
  position: [number, number, number];
  velocity: [number, number, number];
  include_area: [number, number, number, number, number, number];
  target_pos: [number, number, number];
};

export type RawHumanData = {
  hid: string;
  ts: number;
  position: [number, number, number];
  velocity: [number, number, number];
};

export type RawBuildingData = {
  id: string;
  bbox: [number, number, number, number, number, number];
};

export type RawData = {
  devices?: RawDeviceData[];
  humans?: RawHumanData[];
  buildings?: RawBuildingData[];
};
