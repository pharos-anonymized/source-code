import { useAnimations, useGLTF } from "@react-three/drei";
import React, { useEffect, useMemo, useRef } from "react";
import { Group, LoopRepeat } from "three";
import { SkeletonUtils } from "three-stdlib";

export type BusinessManProps = React.ComponentProps<"group"> & {
  walking?: boolean;
};

export function BusinessMan({ walking = false, ...props }: BusinessManProps) {
  const { scene, animations } = useGLTF("/models/business-man/scene.gltf");
  const clone = useMemo(() => SkeletonUtils.clone(scene), [scene]);

  const ref = useRef<Group>(null!);
  const { actions } = useAnimations(animations, ref);

  // Play the appropriate animation based on the walking state
  useEffect(() => {
    const newAction = walking ? actions["Rig|walk"] : actions["Rig|idle"];
    const prevAction = walking ? actions["Rig|idle"] : actions["Rig|walk"];
    if (!newAction || !prevAction) return;
    prevAction.stop();
    newAction.reset().setLoop(LoopRepeat, Infinity).play();
  }, [actions, walking]);

  return <primitive ref={ref} object={clone} {...props} />;
}

useGLTF.preload("/models/business-man/scene.gltf");
