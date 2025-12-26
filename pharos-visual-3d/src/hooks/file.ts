import { useEventListener } from "usehooks-ts";
import { useState } from "react";

export const useFileDrop = () => {
  const [file, setFile] = useState<File | null>(null);
  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0];
    if (file) setFile(file);
  };
  useEventListener("drop", handleDrop);
  useEventListener("dragover", (e) => e.preventDefault());

  return { file };
};
