import { GripVertical, X } from "lucide-react";
import type { DragEvent, JSX, ReactNode } from "react";
import { useState } from "react";

import { WorkbenchIconAction } from "@/components/workbench";
import { cn } from "@/lib/utils";

import type { DropPosition } from "./useDashboardLayout";

interface DashboardPanelProps {
  id: string;
  title?: string;
  description?: string;
  children: ReactNode;
  onReorder: (activeId: string, overId: string, position: DropPosition) => void;
  onRemove: (id: string) => void;
  /** Skip the chrome when the child is already a full surface (e.g. KPI strip). */
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
        "flex items-center gap-1 opacity-0 transition-opacity",
        "group-hover/panel:opacity-100 group-focus-within/panel:opacity-100",
        bare && "absolute right-2 top-2 z-10",
      )}
    >
      <WorkbenchIconAction
        label={title ? `Drag ${title} panel` : "Drag panel"}
        onMouseDown={() => setDraggable(true)}
        onMouseUp={() => setDraggable(false)}
        onTouchStart={() => setDraggable(true)}
        onTouchEnd={() => setDraggable(false)}
        className="cursor-grab text-muted-foreground active:cursor-grabbing"
      >
        <GripVertical className="h-3.5 w-3.5" />
      </WorkbenchIconAction>
      <WorkbenchIconAction
        label={title ? `Remove ${title} panel` : "Remove panel"}
        kind="danger"
        onClick={() => onRemove(id)}
        className="text-muted-foreground hover:bg-status-failed-soft hover:text-status-failed-foreground"
      >
        <X className="h-3.5 w-3.5" />
      </WorkbenchIconAction>
    </div>
  );

  return (
    <section
      aria-label={title ? `${title} dashboard panel` : "Dashboard panel"}
      data-panel-id={id}
      draggable={draggable}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "group/panel relative flex h-full min-h-0 flex-col overflow-hidden rounded-panel border border-border bg-surface",
        bare && "border-transparent bg-transparent",
        isDragging && "opacity-50",
      )}
    >
      {dropPosition && !isDragging && <DropIndicator position={dropPosition} />}
      {!bare && (title || description) ? (
        <header className="flex flex-row items-start justify-between gap-3 border-b border-border px-3 py-2">
          <div className="min-w-0 space-y-1">
            {title && (
              <h3 className="text-body font-medium leading-none text-foreground">{title}</h3>
            )}
            {description && <p className="text-label text-muted-foreground">{description}</p>}
          </div>
          {controls}
        </header>
      ) : (
        controls
      )}
      <div className={cn("min-h-0 flex-1", bare ? "p-0" : "p-3")}>{children}</div>
    </section>
  );
};

interface DropIndicatorProps {
  position: DropPosition;
}

const DropIndicator = ({ position }: DropIndicatorProps): JSX.Element => (
  <div
    className={cn(
      "mol-motion-enter-fade pointer-events-none absolute z-20 bg-accent/70",
      position === "left" && "left-0 top-0 h-full w-1",
      position === "right" && "right-0 top-0 h-full w-1",
      position === "top" && "left-0 top-0 h-1 w-full",
      position === "bottom" && "bottom-0 left-0 h-1 w-full",
    )}
  />
);
