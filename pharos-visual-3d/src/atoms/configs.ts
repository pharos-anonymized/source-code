import { atom } from "jotai";
import { atomWithStorage } from "jotai/utils";

interface Config {
  showStats: boolean;
  showGrid: boolean;
  showHumanVelocity: boolean;
  showDeviceTarget: boolean;
  worldMinX: number;
  worldMinZ: number;
  worldMaxX: number;
  worldMaxZ: number;
}

const defaultConfig: Config = {
  showStats: false,
  showGrid: true,
  showHumanVelocity: false,
  showDeviceTarget: true,
  worldMinX: -5,
  worldMinZ: -5,
  worldMaxX: 35,
  worldMaxZ: 35,
};

export const showConfigPanelAtom = atom(false);
export const configAtom = atomWithStorage("APP_CONFIG", defaultConfig);

const configPropertyAtom = <K extends keyof Config>(key: K) => {
  return atom(
    (get) => get(configAtom)[key],
    (get, set, value: Config[K]) => {
      const config = get(configAtom);
      set(configAtom, { ...config, [key]: value });
    }
  );
};

export const showStatsAtom = configPropertyAtom("showStats");
export const showGridAtom = configPropertyAtom("showGrid");
export const showHumanVelocityAtom = configPropertyAtom("showHumanVelocity");
export const showDeviceTargetAtom = configPropertyAtom("showDeviceTarget");
export const worldMinXAtom = configPropertyAtom("worldMinX");
export const worldMinZAtom = configPropertyAtom("worldMinZ");
export const worldMaxXAtom = configPropertyAtom("worldMaxX");
export const worldMaxZAtom = configPropertyAtom("worldMaxZ");

export const worldSizeAtom = atom((get) => {
  const config = get(configAtom);
  return [
    config.worldMaxX - config.worldMinX,
    config.worldMaxZ - config.worldMinZ,
  ];
});

export const worldCenterAtom = atom((get) => {
  const config = get(configAtom);
  return [
    (config.worldMinX + config.worldMaxX) / 2,
    (config.worldMinZ + config.worldMaxZ) / 2,
  ];
});
