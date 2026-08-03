import { ChevronDown, ChevronUp } from "lucide-react";
import type { JSX, KeyboardEvent as ReactKeyboardEvent } from "react";

import { cn } from "@/lib/utils";

import { BOTTOM_PANEL_TABS, type BottomPanelTab } from "./bottomPanelModel";
import { WorkbenchHeartbeat } from "./WorkbenchHeartbeat";

export interface BottomPanelTabStripProps {
  beat: number;
  contextLabel?: string | null;
  isRefreshing: boolean;
  onRefresh?: () => void;
  onTabKeyDown: (event: ReactKeyboardEvent<HTMLButtonElement>, tab: BottomPanelTab) => void;
  open: boolean;
  selectTab: (tab: BottomPanelTab) => void;
  tab: BottomPanelTab;
  toggleOpen: () => void;
}

export const BottomPanelTabStrip = ({
  beat,
  contextLabel,
  isRefreshing,
  onRefresh,
  onTabKeyDown,
  open,
  selectTab,
  tab,
  toggleOpen,
}: BottomPanelTabStripProps): JSX.Element => (
  <div className="flex h-7 flex-none items-center gap-1 border-b border-border/70 bg-background px-1">
    <WorkbenchHeartbeat
      beat={beat}
      label={isRefreshing ? "Syncing…" : "Live sync — click to refresh"}
      running={isRefreshing}
      onClick={onRefresh}
      disabled={isRefreshing || !onRefresh}
    />
    <span className="mx-1 h-3.5 w-px flex-none bg-border" aria-hidden />
    <div className="flex items-center gap-1" role="tablist" aria-label="Bottom panel tabs">
      {BOTTOM_PANEL_TABS.map((item) => {
        const active = tab === item.id;

        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={active}
            aria-controls="workbench-bottom-panel-body"
            tabIndex={active ? 0 : -1}
            id={`bottom-tab-${item.id}`}
            className={cn(
              "h-6 rounded-[var(--radius-control)] px-2 text-label font-medium transition-colors duration-[var(--motion-base)] ease-[var(--motion-ease)] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
              active && open
                ? "bg-interactive text-foreground"
                : "text-muted-foreground hover:bg-interactive/60 hover:text-foreground",
            )}
            onClick={() => selectTab(item.id)}
            onKeyDown={(event) => onTabKeyDown(event, item.id)}
          >
            {item.label}
          </button>
        );
      })}
    </div>
    <div className="min-w-0 flex-1 truncate px-2 font-mono text-micro text-muted-foreground tabular-nums">
      {contextLabel ?? ""}
    </div>
    <button
      type="button"
      className="flex h-6 w-6 flex-none items-center justify-center rounded-[var(--radius-control)] text-muted-foreground hover:bg-interactive hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      onClick={toggleOpen}
      aria-label={open ? "Collapse bottom panel" : "Expand bottom panel"}
      aria-expanded={open}
    >
      {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
    </button>
  </div>
);
