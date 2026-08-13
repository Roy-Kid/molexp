import { AlertTriangle } from "lucide-react";
import { type JSX, useEffect, useMemo, useState } from "react";
import {
  CopyButton,
  EmptyState,
  InventoryCanvas,
  OverviewSurface,
  StatusIcon,
  statusKey,
} from "@/app/components/entity";
import { formatDuration } from "@/app/renderers/dashboardData";
import { workspaceApi } from "@/app/state/api";
import type { RunSummary, WorkflowSummary } from "@/app/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { RunStatusBadge, WorkbenchAction } from "@/components/workbench";
import { normalizeTaskGraph } from "@/components/workflow/flowgram-document";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";
import { parseWorkflowIr, WorkflowGraph } from "@/components/workflow/workflow-graph";
import { formatDateTime } from "@/lib/datetime";
import { cn } from "@/lib/utils";

const formatTimeOfDay = (iso: string | null): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleTimeString();
};

interface RunExecutionsPanelProps {
  run: RunSummary;
  workflow?: WorkflowSummary;
  selectedExecutionId: string | null;
  onSelectExecution: (executionId: string) => void;
  onInspectTask: (taskId: string, runId: string) => void;
  onOpenWorkflow?: () => void;
  onViewLogs?: () => void;
}

/**
 * Attempt history + per-attempt workflow graph.
 * Layout matches Overview language: padded canvas, shadcn Table (no card lists).
 */
export const RunExecutionsPanel = ({
  run,
  workflow,
  selectedExecutionId,
  onSelectExecution,
  onInspectTask,
  onOpenWorkflow,
  onViewLogs,
}: RunExecutionsPanelProps): JSX.Element => {
  const history = run.executionHistory;
  const [executionGraph, setExecutionGraph] = useState<TaskGraphJson | null>(null);
  const [executionGraphError, setExecutionGraphError] = useState<string | null>(null);

  const effectiveExecutionId =
    selectedExecutionId ?? history[history.length - 1]?.executionId ?? null;
  const effectiveExecution = history.find((rec) => rec.executionId === effectiveExecutionId);
  const effectiveIndex = effectiveExecution
    ? history.findIndex((rec) => rec.executionId === effectiveExecution.executionId)
    : -1;
  const shouldPoll =
    run.status === "running" ||
    effectiveExecution?.status === "running" ||
    effectiveExecution?.finishedAt === null;

  useEffect(() => {
    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    const load = (): void => {
      if (!effectiveExecutionId) {
        setExecutionGraph(null);
        setExecutionGraphError(null);
        return;
      }
      workspaceApi
        .getRunExecution(run.projectId, run.experimentId, run.id, effectiveExecutionId)
        .then((response) => {
          if (cancelled) return;
          if (response.workflow) {
            setExecutionGraph(normalizeTaskGraph(response.workflow));
          } else {
            setExecutionGraph(null);
          }
          setExecutionGraphError(null);
        })
        .catch((err) => {
          if (cancelled) return;
          setExecutionGraphError(
            err instanceof Error ? err.message : "Failed to load workflow execution",
          );
        });
    };

    load();
    if (shouldPoll) {
      interval = setInterval(load, 1000);
    }
    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, [effectiveExecutionId, run.experimentId, run.id, run.projectId, shouldPoll]);

  const staticWorkflowIr = useMemo(() => parseWorkflowIr(run.workflowSource), [run.workflowSource]);
  const workflowIr = executionGraph ?? staticWorkflowIr;
  const failedTasks = executionGraph
    ? executionGraph.task_configs.filter((t) => statusKey(t.status) === "failed")
    : [];
  const attemptFailed =
    statusKey(effectiveExecution?.status) === "failed" || statusKey(run.status) === "failed";

  if (history.length === 0) {
    return (
      <OverviewSurface>
        <InventoryCanvas>
          <EmptyState
            title="No executions"
            description={
              run.status === "pending"
                ? "Start the run to open the first execution attempt."
                : "This run has no recorded attempts."
            }
          />
        </InventoryCanvas>
      </OverviewSurface>
    );
  }

  return (
    <OverviewSurface>
      <InventoryCanvas className="max-w-6xl space-y-8">
        <section className="space-y-3">
          <h3 className="text-body-lg font-medium text-foreground">
            Attempts
            <span className="ml-2 font-mono text-micro font-normal text-muted-foreground">
              {history.length}
            </span>
          </h3>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-16">#</TableHead>
                <TableHead className="w-24">State</TableHead>
                <TableHead>Execution</TableHead>
                <TableHead className="w-40">Started</TableHead>
                <TableHead className="w-28">Duration</TableHead>
                <TableHead className="w-36">Backend</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {history.map((rec, index) => {
                const active = rec.executionId === effectiveExecutionId;
                const d = formatDuration(rec.startedAt, rec.finishedAt);
                return (
                  <TableRow
                    key={rec.executionId}
                    className={cn("cursor-pointer", active && "bg-muted/40")}
                    onClick={() => onSelectExecution(rec.executionId)}
                    data-state={active ? "selected" : undefined}
                  >
                    <TableCell className="font-mono text-label text-muted-foreground">
                      <span className="inline-flex items-center gap-2">
                        <StatusIcon status={rec.status} />
                        {index + 1}
                      </span>
                    </TableCell>
                    <TableCell>
                      <RunStatusBadge status={rec.status} size="sm" />
                    </TableCell>
                    <TableCell className="font-mono text-label text-foreground">
                      {rec.executionId}
                    </TableCell>
                    <TableCell className="text-label text-muted-foreground">
                      {formatTimeOfDay(rec.startedAt)}
                    </TableCell>
                    <TableCell className="font-mono text-label text-muted-foreground">
                      {d ?? "—"}
                    </TableCell>
                    <TableCell className="truncate font-mono text-label text-muted-foreground">
                      {rec.schedulerJobId ?? run.executorInfo.backend ?? "local"}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </section>

        {effectiveExecution && (
          <section className="space-y-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-body-lg font-medium text-foreground">
                Execution #{effectiveIndex + 1}
              </h3>
              <div className="flex items-center gap-3">
                <CopyButton value={effectiveExecution.executionId} label="execution ID" />
                {onViewLogs && (
                  <WorkbenchAction kind="ghost" size="compact" type="button" onClick={onViewLogs}>
                    Logs
                  </WorkbenchAction>
                )}
                {workflow && onOpenWorkflow && (
                  <WorkbenchAction
                    kind="ghost"
                    size="compact"
                    type="button"
                    onClick={onOpenWorkflow}
                  >
                    Open workflow
                  </WorkbenchAction>
                )}
              </div>
            </div>
            <Table>
              <TableBody>
                <TableRow>
                  <TableCell className="w-36 text-label text-muted-foreground">State</TableCell>
                  <TableCell>
                    <RunStatusBadge status={effectiveExecution.status} size="sm" />
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="text-label text-muted-foreground">Start</TableCell>
                  <TableCell className="font-mono text-label">
                    {formatDateTime(effectiveExecution.startedAt)}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="text-label text-muted-foreground">End</TableCell>
                  <TableCell className="font-mono text-label">
                    {formatDateTime(effectiveExecution.finishedAt)}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="text-label text-muted-foreground">Duration</TableCell>
                  <TableCell className="font-mono text-label">
                    {formatDuration(effectiveExecution.startedAt, effectiveExecution.finishedAt) ??
                      "—"}
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell className="text-label text-muted-foreground">Backend</TableCell>
                  <TableCell className="font-mono text-label">
                    {effectiveExecution.schedulerJobId ?? run.executorInfo.backend ?? "local"}
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </section>
        )}

        <section className="space-y-3">
          <h3 className="text-body-lg font-medium text-foreground">Attempt workflow</h3>
          {workflowIr ? (
            <>
              {failedTasks.length > 0 ? (
                <div className="rounded-panel border border-status-failed/25 bg-status-failed-soft px-3 py-3 text-label">
                  <div className="flex flex-wrap items-center gap-2 font-medium text-status-failed-foreground">
                    <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                    Failed at{" "}
                    {failedTasks.map((t, i) => (
                      <span key={t.id}>
                        {i > 0 && ", "}
                        <WorkbenchAction
                          kind="ghost"
                          size="content"
                          type="button"
                          className="font-mono underline-offset-2 hover:underline"
                          onClick={() => onInspectTask(t.id, run.id)}
                        >
                          {t.id}
                        </WorkbenchAction>
                      </span>
                    ))}
                  </div>
                </div>
              ) : attemptFailed && !executionGraph ? (
                <div className="rounded-panel border border-status-failed/25 bg-status-failed-soft px-3 py-3 text-label text-status-failed-foreground">
                  This attempt failed before any task ran.
                </div>
              ) : null}
              <div className="overflow-hidden rounded-panel border border-border bg-canvas">
                <WorkflowGraph
                  ir={workflowIr}
                  height={420}
                  onNodeClick={(taskId) => onInspectTask(taskId, run.id)}
                />
              </div>
              {executionGraphError && (
                <p className="text-label text-destructive">{executionGraphError}</p>
              )}
            </>
          ) : (
            <p className="text-label text-muted-foreground">
              No workflow snapshot recorded for this run.
            </p>
          )}
        </section>
      </InventoryCanvas>
    </OverviewSurface>
  );
};
