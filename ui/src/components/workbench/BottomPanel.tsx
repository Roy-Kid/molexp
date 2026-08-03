/**
 * Workbench bottom region — Logs · Problems · Runs · Artifacts.
 *
 * Feature bodies plug in through `slots`; persistence, resize behavior, and
 * tab keyboard behavior live in focused support modules.
 */

import type { JSX, ReactNode } from "react";

import { useSyncPulse } from "@/app/state/syncPulse";
import { cn } from "@/lib/utils";

import { BottomPanelEmptyState } from "./BottomPanelEmptyState";
import { BottomPanelResizeHandle } from "./BottomPanelResizeHandle";
import { BottomPanelTabStrip } from "./BottomPanelTabStrip";
import type { BottomPanelTab } from "./bottomPanelModel";
import { useBottomPanelState } from "./useBottomPanelState";

export type { BottomPanelTab } from "./bottomPanelModel";

export interface BottomPanelSlotProps {
  slots?: Partial<Record<BottomPanelTab, ReactNode>>;
  contextLabel?: string | null;
  className?: string;
  /** Manual refresh (wired to the heartbeat control). */
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

export const BottomPanel = ({
  slots,
  contextLabel,
  className,
  onRefresh,
  isRefreshing = false,
}: BottomPanelSlotProps): JSX.Element => {
  const beat = useSyncPulse();
  const panel = useBottomPanelState();
  const body = slots?.[panel.tab] ?? <BottomPanelEmptyState tab={panel.tab} />;

  return (
    <section
      aria-label="Workbench bottom panel"
      className={cn(
        "mol-motion-panel flex min-h-0 flex-none flex-col border-t border-border bg-surface",
        className,
      )}
      data-open={panel.open ? "true" : "false"}
      data-resizing={panel.resizing ? "true" : "false"}
      style={{ height: panel.open ? panel.height : "var(--spacing-statusbar)" }}
    >
      {panel.open && (
        <BottomPanelResizeHandle
          height={panel.height}
          maximumHeight={panel.maximumHeight}
          onKeyDown={panel.onResizeKeyDown}
          onPointerDown={panel.onResizePointerDown}
          onPointerMove={panel.onResizePointerMove}
          onPointerUp={panel.onResizePointerUp}
        />
      )}
      <BottomPanelTabStrip
        beat={beat}
        contextLabel={contextLabel}
        isRefreshing={isRefreshing}
        onRefresh={onRefresh}
        onTabKeyDown={panel.onTabKeyDown}
        open={panel.open}
        selectTab={panel.selectTab}
        tab={panel.tab}
        toggleOpen={panel.toggleOpen}
      />
      {panel.open && (
        <div
          id="workbench-bottom-panel-body"
          role="tabpanel"
          aria-labelledby={`bottom-tab-${panel.tab}`}
          className="mol-motion-enter-from-bottom min-h-0 flex-1 overflow-auto"
        >
          {body}
        </div>
      )}
    </section>
  );
};
