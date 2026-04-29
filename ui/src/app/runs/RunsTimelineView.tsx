import type { JSX } from "react";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { RunsGanttChart } from "./RunsGanttChart";
import type { WorkspaceExecutionRow, WorkspaceRunRow } from "./types";

export type GanttMode = "runs" | "executions";

interface RunsTimelineViewProps {
  rows: WorkspaceRunRow[];
  mode: GanttMode;
  onModeChange: (mode: GanttMode) => void;
  onSelectRun: (run: WorkspaceRunRow) => void;
  onSelectExecution: (run: WorkspaceRunRow, execution: WorkspaceExecutionRow) => void;
}

export const RunsTimelineView = ({
  rows,
  mode,
  onModeChange,
  onSelectRun,
  onSelectExecution,
}: RunsTimelineViewProps): JSX.Element => (
  <div className="flex h-full min-h-0 flex-col gap-2 rounded-md border border-border/60 bg-card p-3">
    <div className="flex items-center justify-between gap-3">
      <div>
        <h3 className="text-sm font-semibold text-foreground">Run timeline</h3>
        <p className="text-[11px] text-muted-foreground">
          Click a bar to load the run in the inspector. Faded bars are queued / pending.
        </p>
      </div>
      <Tabs value={mode} onValueChange={(next) => onModeChange(next as GanttMode)}>
        <TabsList className="h-7 p-0.5">
          <TabsTrigger value="runs" className="h-6 px-2 text-[11px] uppercase tracking-wide">
            By runs
          </TabsTrigger>
          <TabsTrigger value="executions" className="h-6 px-2 text-[11px] uppercase tracking-wide">
            By executions
          </TabsTrigger>
        </TabsList>
      </Tabs>
    </div>
    <div className="min-h-0 flex-1">
      <RunsGanttChart
        rows={rows}
        mode={mode}
        onSelectRun={onSelectRun}
        onSelectExecution={onSelectExecution}
      />
    </div>
  </div>
);
