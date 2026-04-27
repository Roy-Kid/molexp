import { GripVertical, X } from "lucide-react";
import { useState } from "react";
import type { DragEvent, JSX, ReactNode } from "react";

import { cn } from "@/lib/utils";

interface DashboardPanelProps {
  id: string;
  title?: string;
  children: ReactNode;
  onReorder: (activeId: string, overId: string) => void;
  onRemove: (id: string) => void;
}

const DRAG_MIME = "application/x-molexp-panel-id";

export const DashboardPanel = ({
  id,
  title,
  children,
  onReorder,
  onRemove,
}: DashboardPanelProps): JSX.Element => {
  const [draggable, setDraggable] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragStart = (event: DragEvent<HTMLDivElement>): void => {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData(DRAG_MIME, id);
    setIsDragging(true);
  };

  const handleDragEnd = (): void => {
    setDraggable(false);
    setIsDragging(false);
    setIsDragOver(false);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>): void => {
    if (!event.dataTransfer.types.includes(DRAG_MIME)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    setIsDragOver(true);
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>): void => {
    const next = event.relatedTarget as Node | null;
    if (next && event.currentTarget.contains(next)) return;
    setIsDragOver(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    setIsDragOver(false);
    const activeId = event.dataTransfer.getData(DRAG_MIME);
    if (activeId && activeId !== id) onReorder(activeId, id);
  };

  return (
    <div
      data-panel-id={id}
      draggable={draggable}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "group/panel relative rounded border border-transparent transition-all",
        isDragOver && !isDragging && "border-primary/60 bg-primary/5",
        isDragging && "opacity-50",
      )}
    >
      <div className="absolute right-1 top-1 z-10 flex items-center gap-0.5 opacity-0 transition-opacity group-hover/panel:opacity-100 focus-within:opacity-100">
        <button
          type="button"
          aria-label={title ? `Drag ${title} panel` : "Drag panel"}
          title="Drag to reorder"
          onMouseDown={() => setDraggable(true)}
          onMouseUp={() => setDraggable(false)}
          onTouchStart={() => setDraggable(true)}
          onTouchEnd={() => setDraggable(false)}
          className="flex h-6 w-6 cursor-grab items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground active:cursor-grabbing"
        >
          <GripVertical className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          aria-label={title ? `Remove ${title} panel` : "Remove panel"}
          title="Remove panel"
          onClick={() => onRemove(id)}
          className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {children}
    </div>
  );
};
