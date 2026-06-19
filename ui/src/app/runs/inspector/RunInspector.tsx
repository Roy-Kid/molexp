import { ExternalLink, Inbox, X } from "lucide-react";
import type { JSX } from "react";
import { useState } from "react";

import { StatusBadge } from "@/app/components/entity";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { RunMetricsView } from "../metrics/RunMetricsView";
import type { WorkspaceRunRow } from "../types";
import { RunInspectorDetails } from "./RunInspectorDetails";
import { RunInspectorPlaceholder } from "./RunInspectorPlaceholder";

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

  if (!run) {
    return (
      <aside className="flex h-full w-[320px] shrink-0 flex-col border-l border-border bg-card">
        <header className="flex items-center justify-between border-b border-border px-4 py-2.5">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Inspector
          </span>
        </header>
        <div className="flex flex-1 flex-col items-center justify-center gap-2 px-6 text-center text-xs text-muted-foreground">
          <Inbox className="h-5 w-5 opacity-50" />
          <p>No run selected.</p>
          <p className="text-[11px]">
            Click a row in the Jobs table or a bar in the Timeline to inspect details, attempts and
            logs.
          </p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-[320px] shrink-0 flex-col border-l border-border bg-card">
      <header className="flex items-start justify-between gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <StatusBadge status={run.status} size="sm" dot />
            <p
              className="min-w-0 truncate font-mono text-sm font-medium text-foreground"
              title={run.id}
            >
              {run.name || run.id}
            </p>
          </div>
          <p className="mt-1 truncate text-[11px] text-muted-foreground">
            {run.projectName} · {run.experimentName}
          </p>
          <p
            className="mt-0.5 truncate font-mono text-[10px] text-muted-foreground/80"
            title={run.id}
          >
            {run.id}
          </p>
        </div>
        <button
          type="button"
          onClick={onClear}
          className="rounded p-1 text-muted-foreground hover:bg-accent"
          aria-label="Clear selection"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </header>

      <Tabs
        value={tab}
        onValueChange={(next) => setTab(next as InspectorTab)}
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="border-b border-border px-3">
          <TabsList variant="line" className="h-8 px-0">
            <TabsTrigger value="details" className="h-7 px-2 text-xs">
              Details
            </TabsTrigger>
            <TabsTrigger value="logs" className="h-7 px-2 text-xs">
              Logs
            </TabsTrigger>
            <TabsTrigger value="metrics" className="h-7 px-2 text-xs">
              Metrics
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          <TabsContent value="details" className="m-0">
            <RunInspectorDetails
              run={run}
              selectedExecutionId={selectedExecutionId}
              onSelectExecution={onSelectExecution}
            />
          </TabsContent>
          <TabsContent value="logs" className="m-0">
            <RunInspectorPlaceholder
              title="Logs tail not wired yet"
              description={
                <span>
                  Per-attempt log polling lands in the next iteration via the existing{" "}
                  <code className="font-mono">/runs/&lt;id&gt;/executions/&lt;exec&gt;/logs</code>{" "}
                  endpoint. SSE is intentionally not used — the backend does not stream run events.
                </span>
              }
            />
          </TabsContent>
          <TabsContent value="metrics" className="m-0">
            <RunMetricsView
              key={run.id}
              projectId={run.projectId}
              experimentId={run.experimentId}
              runId={run.id}
            />
          </TabsContent>
        </div>
      </Tabs>

      <footer className="border-t border-border px-4 py-2.5">
        <Button
          size="sm"
          variant="outline"
          onClick={() => onOpenRun(run)}
          className="w-full text-xs"
        >
          <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
          Open run detail
        </Button>
      </footer>
    </aside>
  );
};
