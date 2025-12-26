import { showHumanVelocityAtom } from "@/atoms/configs";
import { HumanData } from "@/types";
import { Line, Sphere } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useAtom } from "jotai";
import { ReactNode, useRef, useMemo } from "react";
import { Plane, Vector3, type Group } from "three";

export type HumanModelProps = {
  human: HumanData;
  model: ReactNode;
};

export const HumanModel = ({ human, model }: HumanModelProps) => {
  const { position, velocity } = human;

  const [arrowLength, arrowDirection] = useMemo(() => {
    const length = velocity.length();
    const direction = velocity.clone().normalize();
    return [length, direction];
  }, [velocity]);

  const humanHeight = 1; // approximate half height of a human in meters

  const groupRef = useRef<Group>(null);
  useFrame(() => {
    if (!groupRef.current) return;
    groupRef.current.position.copy(position);
    groupRef.current.rotation.y = Math.atan2(
      arrowDirection.x,
      arrowDirection.z
    );
  });

  const [showHumanVelocity] = useAtom(showHumanVelocityAtom);

  return (
    <group ref={groupRef}>
      {/* Huamn model */}
      {model}
      {/* Arrow of velocity */}
      {showHumanVelocity && (
        <group position={[0, humanHeight, 0]}>
          <Line
            points={[
              [0, 0, 0],
              [0, 0, arrowLength],
            ]}
            color="red"
            lineWidth={2}
          />
          <mesh position={[0, 0, arrowLength]} rotation={[Math.PI / 2, 0, 0]}>
            <coneGeometry args={[0.1, 0.4, 8]} />
            <meshBasicMaterial color="red" />
          </mesh>
        </group>
      )}
      {/* Fear radius */}
      <Sphere args={[5, 32, 32]} position={[0, humanHeight, 0]} visible={true}>
        <meshBasicMaterial
          color="#FFFFAA"
          transparent={true}
          opacity={0.3}
          clippingPlanes={[new Plane(new Vector3(0, 1, 0), 0)]}
        />
      </Sphere>
    </group>
  );
};
