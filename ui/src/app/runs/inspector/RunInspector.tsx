import { ExternalLink, Inbox, X } from "lucide-react";
import type { JSX } from "react";
import { useState } from "react";

import { EmptyState, StatusBadge } from "@/app/components/entity";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

import { RunMetricsView } from "../metrics/RunMetricsView";
import type { WorkspaceRunRow } from "../types";
import { useRunInspectorLogs } from "../useRunInspectorLogs";
import { RunInspectorDetails } from "./RunInspectorDetails";
import { RunInspectorLogs } from "./RunInspectorLogs";

type InspectorTab = "details" | "logs" | "metrics";

interface RunInspectorProps {
  run: WorkspaceRunRow | null;
  selectedExecutionId: string | null;
  onSelectExecution: (id: string | null) => void;
  onClear: () => void;
  onOpenRun: (run: WorkspaceRunRow) => void;
}

export const RunInspector = ({
  run,
  selectedExecutionId,
  onSelectExecution,
  onClear,
  onOpenRun,
}: RunInspectorProps): JSX.Element => {
  const [tab, setTab] = useState<InspectorTab>("details");
  const logsState = useRunInspectorLogs(run, selectedExecutionId, tab === "logs" && run !== null);

  if (!run) {
    return (
      <aside className="flex h-full w-[320px] shrink-0 flex-col border-l border-border/60 bg-card">
        <header className="flex items-center justify-between border-b border-border/60 px-4 py-3">
          <span className="text-sm font-medium text-foreground">Inspector</span>
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
    <aside className="flex h-full w-[320px] shrink-0 flex-col border-l border-border/60 bg-card">
      <header className="flex items-start justify-between gap-2 border-b border-border/60 px-4 py-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex items-center gap-2">
            <StatusBadge status={run.status} size="sm" dot />
            <p
              className="min-w-0 truncate text-sm font-medium tracking-tight text-foreground"
              title={run.id}
            >
              {run.name || run.id}
            </p>
          </div>
          <p className="truncate text-xs text-muted-foreground">
            {run.projectName}
            <span className="mx-1 text-border">·</span>
            {run.experimentName}
          </p>
          <p className="truncate font-mono text-[11px] text-muted-foreground/80" title={run.id}>
            {run.id}
          </p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onClear}
          className="h-7 w-7 shrink-0 text-muted-foreground"
          aria-label="Clear selection"
        >
          <X className="h-3.5 w-3.5" />
        </Button>
      </header>

      <Tabs
        value={tab}
        onValueChange={(next) => setTab(next as InspectorTab)}
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="border-b border-border/60 px-3">
          <TabsList
            variant="line"
            className="h-9 w-full justify-start gap-3 rounded-none bg-transparent p-0"
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
                  "h-9 flex-none rounded-none border-0 border-b-2 border-transparent px-0 text-xs font-medium shadow-none after:hidden",
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

      <footer className="border-t border-border/60 px-4 py-3">
        <Button
          size="sm"
          variant="outline"
          onClick={() => onOpenRun(run)}
          className="h-8 w-full text-xs"
        >
          <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
          Open run detail
        </Button>
      </footer>
    </aside>
  );
};
