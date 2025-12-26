import { useEffect } from "react";
import { Link } from "react-router";
import { DoubleSide } from "three";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { Button, Tooltip } from "@heroui/react";
import { Grid, OrbitControls, Sky, Stats } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { CircleHelp, Settings, Database } from "lucide-react";

import {
  showConfigPanelAtom,
  showGridAtom,
  showStatsAtom,
  worldCenterAtom,
  worldSizeAtom,
} from "@/atoms/configs";
import {
  buildingAtom,
  currentDeviceStateAtom,
  currentHumanStateAtom,
  isPlayingAtom,
  updateDataAtom,
} from "@/atoms/history";
import { BuildingModel } from "@/components/building-model";
import { ConfigPanel } from "@/components/config-panel";
import { DeviceModel } from "@/components/device-model";
import { HumanModel } from "@/components/human-model";
import { TimeController } from "@/components/time-controller";
import { useFileDrop } from "@/hooks/file";
import { useRemoteData } from "@/hooks/remote-data";
import { BusinessMan, DjiTello } from "@/models";
import { readFileAsync } from "@/utils/file";

const Scene = () => {
  const buildings = useAtomValue(buildingAtom);
  const devices = useAtomValue(currentDeviceStateAtom);
  const humans = useAtomValue(currentHumanStateAtom);
  const isPlaying = useAtomValue(isPlayingAtom);

  return (
    <>
      <Sky sunPosition={[100, 100, 100]} />
      <pointLight castShadow position={[100, 100, 100]} intensity={50000} />
      <ambientLight intensity={0.5} />
      <OrbitControls />

      <GroundPlane />

      {devices.map((device) => (
        <DeviceModel key={device.uid} device={device} model={<DjiTello />} />
      ))}

      {humans.map((human) => (
        <HumanModel
          key={human.hid}
          human={human}
          model={<BusinessMan walking={isPlaying} />}
        />
      ))}

      {buildings.map((building, index) => (
        <BuildingModel key={index} building={building} />
      ))}
    </>
  );
};

const GroundPlane = () => {
  const [showGrid] = useAtom(showGridAtom);
  const [sizeX, sizeY] = useAtomValue(worldSizeAtom);
  const [centerX, centerY] = useAtomValue(worldCenterAtom);

  return (
    <>
      {showGrid && (
        <Grid
          position={[centerX, 0, centerY]}
          args={[sizeX, sizeY]}
          cellSize={1}
          sectionSize={10}
        />
      )}
      <mesh
        rotation={[Math.PI * -0.5, 0, 0]}
        position={[centerX, -0.01, centerY]}
      >
        <planeGeometry args={[sizeX, sizeY]} />
        <meshStandardMaterial side={DoubleSide} />
      </mesh>
    </>
  );
};

const UIOverlay = () => {
  const [showConfigPanel, setShowConfigPanel] = useAtom(showConfigPanelAtom);
  const [showStats] = useAtom(showStatsAtom);
  const handleDataChange = useSetAtom(updateDataAtom);

  const loadDemoData = async () => {
    const response = await fetch("/vis-demo.json");
    const demoData = await response.json();
    handleDataChange(demoData);
  };

  const toggleConfigPanel = () => setShowConfigPanel(!showConfigPanel);

  return (
    <>
      <section className="top-4 right-4 absolute flex gap-2">
        <Tooltip content="Load Demo Data">
          <Button isIconOnly onPress={loadDemoData}>
            <Database className="text-gray-600" />
          </Button>
        </Tooltip>
        <Tooltip content="Settings">
          <Button isIconOnly onPress={toggleConfigPanel}>
            <Settings className="text-gray-600" />
          </Button>
        </Tooltip>
        <Link to="/help">
          <Tooltip content="Help Page">
            <Button isIconOnly>
              <CircleHelp className="text-gray-600" />
            </Button>
          </Tooltip>
        </Link>
        {showStats && <Stats className="mt-2 ml-2" />}
      </section>
      <ConfigPanel />
      <section className="absolute bottom-0 w-full">
        <TimeController />
      </section>
    </>
  );
};

export const HomePage = () => {
  const handleDataChange = useSetAtom(updateDataAtom);

  // Handle file drop
  const { file } = useFileDrop();
  useEffect(() => {
    if (!file) return;
    readFileAsync(file).then((data) => handleDataChange(JSON.parse(data)));
  }, [file, handleDataChange]);

  // Handle remote data change
  const { data: remoteData } = useRemoteData({ enabled: !file });
  useEffect(() => {
    if (!remoteData) return;
    handleDataChange(remoteData);
  }, [remoteData, handleDataChange]);

  return (
    <div className="w-full h-screen">
      <Canvas
        camera={{ position: [40, 20, 40] }}
        onCreated={({ gl }) => {
          gl.localClippingEnabled = true;
        }}
      >
        <Scene />
      </Canvas>
      <UIOverlay />
    </div>
  );
};
