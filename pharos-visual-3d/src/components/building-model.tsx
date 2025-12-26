import { BuildingData } from "@/types";
import { ThreeElements } from "@react-three/fiber";
import { useRef } from "react";
import { Mesh, type Group, type MeshLambertMaterialParameters } from "three";

export type BuildingModelProps = {
  building: BuildingData;
  meshProps?: ThreeElements["mesh"];
  materialProps?: MeshLambertMaterialParameters;
};

export const BuildingModel = ({
  building,
  meshProps,
  materialProps,
}: BuildingModelProps) => {
  const { bbox } = building;
  const meshRef = useRef<Mesh>(null);
  const groupRef = useRef<Group>(null);

  return (
    <group ref={groupRef} renderOrder={-1}>
      <mesh {...meshProps} ref={meshRef} position={bbox.position}>
        <boxGeometry args={bbox.size.toArray()} />
        <meshLambertMaterial
          transparent
          opacity={1}
          color={materialProps?.color || "#b0b0b0"}
          {...materialProps}
        />
      </mesh>
    </group>
  );
};
