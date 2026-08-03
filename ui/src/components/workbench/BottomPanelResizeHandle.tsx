import type { JSX, KeyboardEventHandler, PointerEventHandler } from "react";

import { BOTTOM_PANEL_MIN_HEIGHT } from "./bottomPanelModel";

export interface BottomPanelResizeHandleProps {
  height: number;
  maximumHeight: number;
  onKeyDown: KeyboardEventHandler<HTMLHRElement>;
  onPointerDown: PointerEventHandler<HTMLHRElement>;
  onPointerMove: PointerEventHandler<HTMLHRElement>;
  onPointerUp: PointerEventHandler<HTMLHRElement>;
}

export const BottomPanelResizeHandle = ({
  height,
  maximumHeight,
  onKeyDown,
  onPointerDown,
  onPointerMove,
  onPointerUp,
}: BottomPanelResizeHandleProps): JSX.Element => (
  <hr
    aria-orientation="horizontal"
    aria-label="Resize bottom panel"
    aria-controls="workbench-bottom-panel-body"
    aria-valuemin={BOTTOM_PANEL_MIN_HEIGHT}
    aria-valuemax={maximumHeight}
    aria-valuenow={height}
    aria-valuetext={`${height} pixels`}
    tabIndex={0}
    className="m-0 flex h-1.5 flex-none cursor-row-resize items-center justify-center border-0 bg-surface-subtle after:block after:h-0.5 after:w-8 after:rounded-full after:bg-border hover:bg-interactive focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
    onPointerDown={onPointerDown}
    onPointerMove={onPointerMove}
    onPointerUp={onPointerUp}
    onPointerCancel={onPointerUp}
    onKeyDown={onKeyDown}
  />
);
