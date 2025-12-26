import {
  showGridAtom,
  showStatsAtom,
  showConfigPanelAtom,
  worldMinXAtom,
  worldMinZAtom,
  worldMaxXAtom,
  worldMaxZAtom,
  showHumanVelocityAtom,
  showDeviceTargetAtom,
} from "@/atoms/configs";
import { cn, NumberInput } from "@heroui/react";
import { Card, CardBody, Switch } from "@heroui/react";
import { useAtom } from "jotai";

export const ConfigPanel = () => {
  const [showConfigPanel] = useAtom(showConfigPanelAtom);

  const [showStats, setShowStats] = useAtom(showStatsAtom);
  const [showGrid, setShowGrid] = useAtom(showGridAtom);
  const [showHumanVelocity, setShowHumanVelocity] = useAtom(
    showHumanVelocityAtom
  );
  const [showDeviceTarget, setShowDeviceTarget] = useAtom(showDeviceTargetAtom);

  const [worldMinX, setWorldMinX] = useAtom(worldMinXAtom);
  const [worldMinZ, setWorldMinZ] = useAtom(worldMinZAtom);
  const [worldMaxX, setWorldMaxX] = useAtom(worldMaxXAtom);
  const [worldMaxZ, setWorldMaxZ] = useAtom(worldMaxZAtom);

  return (
    <div
      className={cn(
        "absolute top-16 right-4 z-10 transition",
        showConfigPanel ? "opacity-100" : "opacity-0"
      )}
    >
      <Card className="w-120 p-4 backdrop-blur-sm bg-white/30">
        <CardBody>
          <h2 className="text-xl font-semibold mb-4">Config Panel</h2>
          <div className="grid grid-cols-2 gap-4">
            <Switch isSelected={showStats} onValueChange={setShowStats}>
              Show Stats
            </Switch>
            <Switch isSelected={showGrid} onValueChange={setShowGrid}>
              Show Grid
            </Switch>
            <Switch
              isSelected={showHumanVelocity}
              onValueChange={setShowHumanVelocity}
            >
              Show Human Velocity
            </Switch>
            <Switch
              isSelected={showDeviceTarget}
              onValueChange={setShowDeviceTarget}
            >
              Show Device Target
            </Switch>
            <section className="col-span-2 space-y-2">
              <label>World Bounds</label>
              <div className="grid grid-cols-2 gap-4">
                <NumberInput
                  label="Min X"
                  value={worldMinX}
                  onValueChange={setWorldMinX}
                />
                <NumberInput
                  label="Min Z"
                  value={worldMinZ}
                  onValueChange={setWorldMinZ}
                />
                <NumberInput
                  label="Max X"
                  value={worldMaxX}
                  onValueChange={setWorldMaxX}
                />
                <NumberInput
                  label="Max Z"
                  value={worldMaxZ}
                  onValueChange={setWorldMaxZ}
                />
              </div>
            </section>
          </div>
        </CardBody>
      </Card>
    </div>
  );
};
