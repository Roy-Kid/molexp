import { GripVertical, X } from "lucide-react";
import type { DragEvent, JSX, ReactNode } from "react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

import type { DropPosition } from "./useDashboardLayout";

interface DashboardPanelProps {
  id: string;
  title?: string;
  description?: string;
  children: ReactNode;
  onReorder: (activeId: string, overId: string, position: DropPosition) => void;
  onRemove: (id: string) => void;
  /** Skip the card chrome when the child is already a full surface (e.g. KPI strip). */
  bare?: boolean;
}

const DRAG_MIME = "application/x-molexp-panel-id";

const computeDropPosition = (event: DragEvent<HTMLDivElement>): DropPosition => {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const fx = x / rect.width;
  const fy = y / rect.height;
  const dx = Math.abs(fx - 0.5);
  const dy = Math.abs(fy - 0.5);
  if (dx >= dy) return fx < 0.5 ? "left" : "right";
  return fy < 0.5 ? "top" : "bottom";
};

export const DashboardPanel = ({
  id,
  title,
  description,
  children,
  onReorder,
  onRemove,
  bare = false,
}: DashboardPanelProps): JSX.Element => {
  const [draggable, setDraggable] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [dropPosition, setDropPosition] = useState<DropPosition | null>(null);

  const handleDragStart = (event: DragEvent<HTMLDivElement>): void => {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData(DRAG_MIME, id);
    setIsDragging(true);
  };

  const handleDragEnd = (): void => {
    setDraggable(false);
    setIsDragging(false);
    setDropPosition(null);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>): void => {
    if (!event.dataTransfer.types.includes(DRAG_MIME)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    setDropPosition(computeDropPosition(event));
  };

  const handleDragLeave = (event: DragEvent<HTMLDivElement>): void => {
    const next = event.relatedTarget as Node | null;
    if (next && event.currentTarget.contains(next)) return;
    setDropPosition(null);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>): void => {
    event.preventDefault();
    const activeId = event.dataTransfer.getData(DRAG_MIME);
    const position = computeDropPosition(event);
    setDropPosition(null);
    if (activeId && activeId !== id) onReorder(activeId, id, position);
  };

  const controls = (
    <div
      className={cn(
        "flex items-center gap-0.5 opacity-0 transition-opacity",
        "group-hover/panel:opacity-100 group-focus-within/panel:opacity-100",
        bare && "absolute right-2 top-2 z-10",
      )}
    >
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-label={title ? `Drag ${title} panel` : "Drag panel"}
        title="Drag — drop on left/right to split, top/bottom for new row"
        onMouseDown={() => setDraggable(true)}
        onMouseUp={() => setDraggable(false)}
        onTouchStart={() => setDraggable(true)}
        onTouchEnd={() => setDraggable(false)}
        className="h-7 w-7 cursor-grab text-muted-foreground active:cursor-grabbing"
      >
        <GripVertical className="h-3.5 w-3.5" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-label={title ? `Remove ${title} panel` : "Remove panel"}
        title="Remove panel"
        onClick={() => onRemove(id)}
        className="h-7 w-7 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
      >
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );

  return (
    <Card
      data-panel-id={id}
      draggable={draggable}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "group/panel relative flex h-full min-h-0 flex-col gap-0 overflow-hidden py-0 shadow-none",
        bare && "border-transparent bg-transparent",
        isDragging && "opacity-50",
      )}
    >
      {dropPosition && !isDragging && <DropIndicator position={dropPosition} />}
      {!bare && (title || description) ? (
        <CardHeader className="flex flex-row items-start justify-between gap-3 space-y-0 border-b border-border/60 px-4 py-3">
          <div className="min-w-0 space-y-0.5">
            {title && (
              <CardTitle className="text-sm font-medium leading-none text-foreground">
                {title}
              </CardTitle>
            )}
            {description && (
              <p className="text-xs text-muted-foreground">{description}</p>
            )}
          </div>
          <CardAction className="static col-auto row-auto self-center justify-self-auto">
            {controls}
          </CardAction>
        </CardHeader>
      ) : (
        controls
      )}
      <CardContent className={cn("min-h-0 flex-1", bare ? "p-0" : "p-4")}>{children}</CardContent>
    </Card>
  );
};

interface DropIndicatorProps {
  position: DropPosition;
}

const DropIndicator = ({ position }: DropIndicatorProps): JSX.Element => (
  <div
    className={cn(
      "pointer-events-none absolute z-20 bg-primary/70 transition-all",
      position === "left" && "left-0 top-0 h-full w-1",
      position === "right" && "right-0 top-0 h-full w-1",
      position === "top" && "left-0 top-0 h-1 w-full",
      position === "bottom" && "bottom-0 left-0 h-1 w-full",
    )}
  />
);
