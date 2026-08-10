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
  <section className="flex h-full min-h-0 flex-col" aria-labelledby="runs-timeline-heading">
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border/60 pb-3">
      <div className="min-w-0 space-y-1">
        <h3 id="runs-timeline-heading" className="text-body-lg font-medium text-foreground">
          Run timeline
        </h3>
        <p className="text-label text-muted-foreground">
          Click a bar to load it in the inspector. Faded bars are queued or pending.
        </p>
      </div>
      <Tabs value={mode} onValueChange={(next) => onModeChange(next as GanttMode)}>
        <TabsList className="h-control p-1">
          <TabsTrigger value="runs" className={cn("h-control-compact px-3 text-label font-medium")}>
            By runs
          </TabsTrigger>
          <TabsTrigger
            value="executions"
            className={cn("h-control-compact px-3 text-label font-medium")}
          >
            By executions
          </TabsTrigger>
        </TabsList>
      </Tabs>
    </div>
    <div className="min-h-0 flex-1 pt-3">
      <RunsGanttChart
        rows={rows}
        mode={mode}
        onSelectRun={onSelectRun}
        onSelectExecution={onSelectExecution}
      />
    </div>
  </section>
);
