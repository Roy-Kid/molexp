import type { JSX } from "react";

import type { BottomPanelTab } from "./bottomPanelModel";

const EMPTY_COPY: Record<BottomPanelTab, { title: string; detail: string }> = {
  logs: {
    title: "No logs in scope",
    detail: "Select a run to stream stdout and stderr here.",
  },
  problems: {
    title: "No problems",
    detail: "Validation and compile issues for the active graph will list here.",
  },
  runs: {
    title: "No recent runs",
    detail: "Workspace run activity will appear in this tab.",
  },
  artifacts: {
    title: "No artifacts",
    detail: "Select a run to browse its artifacts without leaving the graph.",
  },
};

export const BottomPanelEmptyState = ({ tab }: { tab: BottomPanelTab }): JSX.Element => {
  const copy = EMPTY_COPY[tab];

  return (
    <div className="flex h-full min-h-0 flex-col items-start justify-center gap-1 px-3 py-2 text-body text-muted-foreground">
      <p className="font-medium text-foreground/80">{copy.title}</p>
      <p className="text-label">{copy.detail}</p>
    </div>
  );
};
