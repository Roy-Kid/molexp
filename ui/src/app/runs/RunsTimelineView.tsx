import type { JSX } from "react";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

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
  <div className="flex h-full min-h-0 flex-col gap-3 rounded-lg border border-border bg-card p-4 shadow-none">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0 space-y-0.5">
        <h3 className="text-sm font-medium text-foreground">Run timeline</h3>
        <p className="text-xs text-muted-foreground">
          Click a bar to load it in the inspector. Faded bars are queued or pending.
        </p>
      </div>
      <Tabs value={mode} onValueChange={(next) => onModeChange(next as GanttMode)}>
        <TabsList className="h-8 p-0.5">
          <TabsTrigger value="runs" className={cn("h-7 px-2.5 text-xs font-medium")}>
            By runs
          </TabsTrigger>
          <TabsTrigger value="executions" className={cn("h-7 px-2.5 text-xs font-medium")}>
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
