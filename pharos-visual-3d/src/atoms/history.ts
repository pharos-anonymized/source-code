import { atom } from "jotai";
import { BuildingData, DeviceData, HumanData, RawData } from "@/types";
import { sortBy } from "lodash-es";
import { toBuildingData, toDeviceData, toHumanData } from "@/utils/data";

// device, human, building data to visualize
export const deviceHistoryAtom = atom<DeviceData[]>([]);
export const humanHistoryAtom = atom<HumanData[]>([]);
export const buildingAtom = atom<BuildingData[]>([]);

// current timestamp of timeline
export const currentTimeAtom = atom(0);

// control the timeline auto play or not
export const isPlayingAtom = atom(false);

// get the min time and max time from the device history data
export const timeRangeAtom = atom((get) => {
  const history = get(deviceHistoryAtom);
  if (!history.length) return [0, 0];
  return [
    Math.min(...history.map((item) => item.timestamp)),
    Math.max(...history.map((item) => item.timestamp)),
  ];
});

// cached sorted history data
const sortedDeviceHistoryAtom = atom((get) => {
  const history = get(deviceHistoryAtom);
  return sortBy(history, "timestamp");
});

const sortedHumanHistoryAtom = atom((get) => {
  const history = get(humanHistoryAtom);
  return sortBy(history, "timestamp");
});

// optimized current state calculation
export const currentDeviceStateAtom = atom((get) => {
  const sortedHistory = get(sortedDeviceHistoryAtom);
  const currentTime = get(currentTimeAtom);

  const latestByUid = new Map<string, DeviceData>();
  for (const item of sortedHistory) {
    if (item.timestamp <= currentTime) {
      latestByUid.set(item.uid, item);
    }
  }

  return Array.from(latestByUid.values());
});

export const currentHumanStateAtom = atom((get) => {
  const sortedHistory = get(sortedHumanHistoryAtom);
  const currentTime = get(currentTimeAtom);

  const latestByHid = new Map<string, HumanData>();
  for (const item of sortedHistory) {
    if (item.timestamp <= currentTime) {
      latestByHid.set(item.hid, item);
    }
  }

  return Array.from(latestByHid.values());
});

// update the device, human, building data from json raw data
export const updateDataAtom = atom(null, (_, set, data: RawData) => {
  const deviceData = toDeviceData(data.devices ?? []);
  set(deviceHistoryAtom, deviceData);
  set(currentTimeAtom, deviceData[0]?.timestamp ?? 0);

  const humanData = toHumanData(data.humans ?? []);
  set(humanHistoryAtom, humanData);

  const buildingData = toBuildingData(data.buildings ?? []);
  set(buildingAtom, buildingData);
});
