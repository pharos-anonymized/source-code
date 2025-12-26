import { currentTimeAtom, isPlayingAtom, timeRangeAtom } from "@/atoms/history";
import { Button, ButtonGroup, Slider } from "@heroui/react";
import { useAtom, useAtomValue } from "jotai";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useEventListener, useInterval } from "usehooks-ts";

export const TimeController = () => {
  const [minTime, maxTime] = useAtomValue(timeRangeAtom);
  const [currentTime, setCurrentTime] = useAtom(currentTimeAtom);
  const [isPlaying, setIsPlaying] = useAtom(isPlayingAtom);

  // ticks of slider, 0.1, 0.2, ..., 0.9
  const ticks = Array.from({ length: 9 }, (_, i) => {
    const percentage = (i + 1) / 10;
    return Math.round(minTime + (maxTime - minTime) * percentage);
  });

  useInterval(() => {
    if (!isPlaying) return;
    if (currentTime >= maxTime) setIsPlaying(false);
    setCurrentTime((time) => Math.min(time + 100, maxTime));
  }, 100);

  const handlePlay = () => {
    setIsPlaying((v) => !v);
    if (currentTime >= maxTime) setCurrentTime(minTime);
  };
  const handlePrev = () => setCurrentTime((t) => Math.max(t - 100, minTime));
  const handleNext = () => setCurrentTime((t) => Math.min(t + 100, maxTime));

  const handleKeyPress = (event: KeyboardEvent) => {
    if (event.code === "Space") handlePlay();
    if (event.code === "ArrowLeft") handlePrev();
    if (event.code === "ArrowRight") handleNext();
  };
  useEventListener("keydown", handleKeyPress);

  return (
    <div className="w-full px-8 py-4">
      <div className="flex items-center gap-4 mb-4">
        <ButtonGroup className="gap-0.5 dark">
          <Button isIconOnly onPress={handlePrev}>
            <SkipBack size={16} />
          </Button>
          <Button isIconOnly onPress={handlePlay}>
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
          </Button>
          <Button isIconOnly onPress={handleNext}>
            <SkipForward size={16} />
          </Button>
        </ButtonGroup>
        <Slider
          className="w-full light mb-0"
          defaultValue={currentTime}
          showOutline={false}
          color="foreground"
          minValue={minTime}
          maxValue={maxTime}
          value={currentTime}
          step={100}
          showTooltip
          onChange={(value) => setCurrentTime(value as number)}
          marks={ticks.map((tick) => ({ value: tick, label: tick.toString() }))}
        />
      </div>
    </div>
  );
};

export default TimeController;
