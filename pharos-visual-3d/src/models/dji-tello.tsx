import { useGLTF } from "@react-three/drei";
import { useMemo } from "react";
import { SkeletonUtils } from "three-stdlib";

export function DjiTello(props: React.ComponentProps<"group">) {
  const { scene } = useGLTF("/models/dji-tello/scene.gltf");
  const clone = useMemo(() => SkeletonUtils.clone(scene), [scene]);

  return <primitive object={clone} {...props} scale={0.25} />;
}

useGLTF.preload("/models/dji-tello/scene.gltf");
