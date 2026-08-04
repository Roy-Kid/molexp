import { ExternalLink, Inbox, X } from "lucide-react";
import type { JSX } from "react";
import { useState } from "react";

import { EmptyState } from "@/app/components/entity";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RunStatusBadge, WorkbenchIconAction } from "@/components/workbench";
import { cn } from "@/lib/utils";
import { RunMetricsView } from "../metrics/RunMetricsView";
import type { WorkspaceRunRow } from "../types";
import { useRunInspectorLogs } from "../useRunInspectorLogs";
import { RunInspectorDetails } from "./RunInspectorDetails";
import { RunInspectorLogs } from "./RunInspectorLogs";

type InspectorTab = "details" | "logs" | "metrics";

export interface RunInspectorProps {
  run: WorkspaceRunRow | null;
  selectedExecutionId: string | null;
  onSelectExecution: (id: string | null) => void;
  onClear: () => void;
  onOpenRun: (run: WorkspaceRunRow) => void;
  className?: string;
}

export type RunInspectorRegistration = Omit<RunInspectorProps, "className">;

export const RunInspector = ({
  run,
  selectedExecutionId,
  onSelectExecution,
  onClear,
  onOpenRun,
  className,
}: RunInspectorProps): JSX.Element => {
  const [tab, setTab] = useState<InspectorTab>("details");
  const logsState = useRunInspectorLogs(run, selectedExecutionId, tab === "logs" && run !== null);

  if (!run) {
    return (
      <aside
        className={cn(
          "flex h-full w-full min-w-0 flex-col border-l border-border/60 bg-card",
          className,
        )}
      >
        <header className="flex items-center justify-between border-b border-border/60 px-4 py-3">
          <span className="text-body-lg font-medium text-foreground">Inspector</span>
        </header>
        <div className="flex flex-1 items-center justify-center px-4">
          <EmptyState
            density="compact"
            icon={<Inbox className="h-5 w-5" />}
            title="No run selected"
            description="Pick a row in Jobs or a bar on Timeline to inspect details, attempts, and logs."
          />
        </div>
      </aside>
    );
  }

  return (
    <aside
      className={cn(
        "flex h-full w-full min-w-0 flex-col border-l border-border/60 bg-card",
        className,
      )}
    >
      <header className="flex items-start justify-between gap-2 border-b border-border/60 px-4 py-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex items-center gap-2">
            <RunStatusBadge status={run.status} size="sm" />
            <p
              className="min-w-0 truncate text-body-lg font-medium tracking-tight text-foreground"
              title={run.id}
            >
              {run.name || run.id}
            </p>
          </div>
          <p className="truncate text-label text-muted-foreground">
            {run.projectName}
            <span className="mx-1 text-border">·</span>
            {run.experimentName}
          </p>
          <p className="truncate font-mono text-micro text-muted-foreground/80" title={run.id}>
            {run.id}
          </p>
        </div>
        <WorkbenchIconAction
          label="Clear selection"
          kind="ghost"
          type="button"
          onClick={onClear}
          className="h-control-compact w-control-compact shrink-0 text-muted-foreground"
          aria-label="Clear selection"
        >
          <X className="h-3.5 w-3.5" />
        </WorkbenchIconAction>
      </header>

      <Tabs
        value={tab}
        onValueChange={(next) => setTab(next as InspectorTab)}
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="border-b border-border/60 px-3">
          <TabsList
            variant="line"
            className="h-control-comfortable w-full justify-start gap-3 rounded-none bg-transparent p-0"
          >
            {(
              [
                ["details", "Details"],
                ["logs", "Logs"],
                ["metrics", "Metrics"],
              ] as const
            ).map(([value, label]) => (
              <TabsTrigger
                key={value}
                value={value}
                className={cn(
                  "h-control-comfortable flex-none rounded-none border-0 border-b border-transparent px-0 text-label font-medium shadow-none after:hidden",
                  "data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none",
                )}
              >
                {label}
              </TabsTrigger>
            ))}
          </TabsList>
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          <TabsContent value="details" className="m-0 h-full overflow-y-auto">
            <RunInspectorDetails
              run={run}
              selectedExecutionId={selectedExecutionId}
              onSelectExecution={onSelectExecution}
            />
          </TabsContent>
          <TabsContent value="logs" className="m-0 flex h-full min-h-0 flex-col overflow-hidden">
            <RunInspectorLogs
              run={run}
              selectedExecutionId={selectedExecutionId}
              onSelectExecution={onSelectExecution}
              logs={logsState.logs}
              error={logsState.error}
              loading={logsState.loading}
              onRefresh={logsState.refresh}
            />
          </TabsContent>
          <TabsContent value="metrics" className="m-0 h-full overflow-y-auto">
            <RunMetricsView
              key={run.id}
              projectId={run.projectId}
              experimentId={run.experimentId}
              runId={run.id}
            />
          </TabsContent>
        </div>
      </Tabs>

      <footer className="flex justify-end border-t border-border/60 px-4 py-3">
        <WorkbenchIconAction label="Open run detail" onClick={() => onOpenRun(run)}>
          <ExternalLink className="h-3.5 w-3.5" />
        </WorkbenchIconAction>
      </footer>
    </aside>
  );
};
