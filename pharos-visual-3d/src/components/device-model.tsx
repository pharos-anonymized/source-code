import { useFrame, ThreeElements } from "@react-three/fiber";
import { Billboard, Line, Text } from "@react-three/drei";
import { useMemo, useRef, ReactNode, Suspense } from "react";
import {
  Box3,
  Mesh,
  type Group,
  type MeshLambertMaterialParameters,
} from "three";
import { DeviceData } from "@/types";
import { useAtom } from "jotai";
import { showDeviceTargetAtom } from "@/atoms/configs";

export type DeviceModelProps = {
  device: DeviceData;
  model: ReactNode;
  meshProps?: ThreeElements["mesh"];
  materialProps?: MeshLambertMaterialParameters;
};

const COLORS = [
  "yellow",
  "green",
  "purple",
  "red",
  "blue",
  "orange",
  "cyan",
  "magenta",
  "lime",
  "pink",
  "teal",
  "brown",
];

const hashCode = (str: string) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash);
};

export const DeviceModel = ({
  device,
  model,
  meshProps,
  materialProps,
}: DeviceModelProps) => {
  const { position, velocity, safeSpace, uid, targetPos } = device;
  const meshRef = useRef<Mesh>(null);
  const groupRef = useRef<Group>(null);

  useFrame(() => {
    if (!groupRef.current) return;
    groupRef.current.position.copy(position);
  });

  const [offset, size] = useMemo(() => {
    const offset = safeSpace.position.clone().sub(position);
    return [offset, safeSpace.size.toArray()];
  }, [safeSpace, position]);

  const delta = useMemo(
    () => targetPos.clone().sub(position),
    [targetPos, position]
  );

  const color = useMemo(
    () => COLORS[hashCode(device.uid) % COLORS.length],
    [device.uid]
  );

  const [showDeviceTarget] = useAtom(showDeviceTargetAtom);

  return (
    <>
      <group ref={groupRef} renderOrder={-2}>
        {model}

        {/* Safe space visualization */}
        <mesh {...meshProps} ref={meshRef} position={offset}>
          <boxGeometry args={size} />
          <meshLambertMaterial
            transparent
            opacity={0.5}
            {...materialProps}
            color={color}
          />
        </mesh>

        {/* Velocity vector */}
        <Line
          points={[0, 0, 0, velocity.x / 2, velocity.y / 2, velocity.z / 2]}
          color={color}
        />

        {/* Device name */}
        <Suspense fallback={null}>
          <Billboard>
            <Text
              renderOrder={-2}
              fontSize={0.4}
              position={[0, safeSpace.maxPoint.y - position.y + 0.5, 0]}
              color={color || "white"}
              font="/fonts/JetBrainsMono-Regular.ttf"
            >
              {uid}
            </Text>
          </Billboard>
        </Suspense>

        {showDeviceTarget && (
          <>
            {/* Target position */}
            <mesh position={delta}>
              <sphereGeometry args={[0.15, 16, 16]} />
              <meshBasicMaterial color={color} />
            </mesh>
            {/* Line to target position */}
            <Line
              points={[[0, 0, 0], delta]}
              color={color}
              lineWidth={1.5}
              dashed
              dashSize={0.5}
              gapSize={0.5}
            />
          </>
        )}
      </group>

      {/* Safe space bounds */}
      <box3Helper
        args={[
          new Box3(device.safeSpace.minPoint, device.safeSpace.maxPoint),
          color,
        ]}
      />
    </>
  );
};
