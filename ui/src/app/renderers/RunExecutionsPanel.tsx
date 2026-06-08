import { ArrowRight, GitBranch } from "lucide-react";
import { type JSX, useEffect, useMemo, useState } from "react";
import { DashboardCard, EmptyState, StatusIcon } from "@/app/components/entity";
import { formatDuration } from "@/app/renderers/dashboardData";
import { workspaceApi } from "@/app/state/api";
import type { RunSummary, WorkflowSummary } from "@/app/types";
import { parseWorkflowIr, WorkflowGraph } from "@/components/workflow/workflow-graph";
import { normalizeTaskGraph } from "@/components/workflow/flowgram-document";
import type { TaskGraphJson } from "@/components/workflow/task-graph-ir";

const formatTimeOfDay = (iso: string | null): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleTimeString();
};

interface RunExecutionsPanelProps {
  run: RunSummary;
  workflow?: WorkflowSummary;
  /** Controlled selected attempt (null → latest). Lifted so the owning viewer
   *  can also scope its log fetch to the chosen attempt. */
  selectedExecutionId: string | null;
  onSelectExecution: (executionId: string) => void;
  onInspectTask: (taskId: string, runId: string) => void;
  onOpenWorkflow?: () => void;
  onViewLogs?: () => void;
}

/**
 * The attempt history of a run plus the per-attempt (read-only) workflow graph.
 * A first-class surface — previously this was buried in an Overview section and
 * absent entirely from the scheduler-backed run view.
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
  }, [
    effectiveExecutionId,
    run.experimentId,
    run.id,
    run.projectId,
    shouldPoll,
  ]);

  const staticWorkflowIr = useMemo(() => parseWorkflowIr(run.workflowSource), [run.workflowSource]);
  const workflowIr = executionGraph ?? staticWorkflowIr;
  const edgeStatusSummary = workflowIr
    ? workflowIr.links.reduce<Record<string, number>>((acc, link) => {
        const status = link.status ?? "pending";
        acc[status] = (acc[status] ?? 0) + 1;
        return acc;
      }, {})
    : {};

  if (history.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState
          icon={<GitBranch className="h-6 w-6" />}
          title="No executions yet"
          description="An execution is recorded each time this run is attempted."
        />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto p-4">
      <ol className="overflow-hidden rounded-lg border border-border/60 bg-card">
        {history.map((rec, index) => {
          const active = rec.executionId === effectiveExecutionId;
          const d = formatDuration(rec.startedAt, rec.finishedAt);
          return (
            <li key={rec.executionId} className="border-b border-border/50 last:border-b-0">
              <button
                type="button"
                onClick={() => onSelectExecution(rec.executionId)}
                className={`grid w-full grid-cols-[auto_70px_minmax(0,1fr)_auto_auto] items-center gap-3 px-3 py-2 text-left text-xs transition-colors ${
                  active ? "bg-muted/50 ring-1 ring-inset ring-foreground/20" : "hover:bg-muted/30"
                }`}
                title={rec.executionId}
              >
                <StatusIcon status={rec.status} />
                <span className="font-mono text-muted-foreground">#{index + 1}</span>
                <span className="truncate font-mono text-[11px] text-foreground">
                  {rec.executionId}
                </span>
                <span className="hidden text-[11px] text-muted-foreground sm:inline">
                  {formatTimeOfDay(rec.startedAt)}
                  {d ? ` · ${d}` : ""}
                </span>
                <span className="max-w-[150px] truncate font-mono text-[11px] text-muted-foreground">
                  {rec.schedulerJobId ?? run.executorInfo.backend ?? "local"}
                </span>
              </button>
            </li>
          );
        })}
      </ol>

      <DashboardCard
        title={effectiveExecution ? `Execution #${effectiveIndex + 1}` : "Execution details"}
        bodyClassName="p-3"
        action={
          onViewLogs && effectiveExecution ? (
            <button
              type="button"
              className="inline-flex items-center gap-1 text-[11px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
              onClick={onViewLogs}
            >
              Logs <ArrowRight className="h-3 w-3" />
            </button>
          ) : undefined
        }
      >
        {effectiveExecution ? (
          <dl className="grid gap-x-4 gap-y-2 text-xs sm:grid-cols-2 lg:grid-cols-5">
            <div className="min-w-0">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">State</dt>
              <dd className="mt-0.5">
                <StatusIcon status={effectiveExecution.status} />
              </dd>
            </div>
            <div className="min-w-0">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">Start</dt>
              <dd className="mt-0.5 truncate text-foreground">{effectiveExecution.startedAt}</dd>
            </div>
            <div className="min-w-0">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">End</dt>
              <dd className="mt-0.5 truncate text-foreground">
                {effectiveExecution.finishedAt ?? "-"}
              </dd>
            </div>
            <div className="min-w-0">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">
                Duration
              </dt>
              <dd className="mt-0.5 font-mono text-foreground">
                {formatDuration(effectiveExecution.startedAt, effectiveExecution.finishedAt) ?? "-"}
              </dd>
            </div>
            <div className="min-w-0">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">Backend</dt>
              <dd className="mt-0.5 truncate font-mono text-foreground">
                {effectiveExecution.schedulerJobId ?? run.executorInfo.backend ?? "local"}
              </dd>
            </div>
          </dl>
        ) : (
          <p className="text-xs italic text-muted-foreground">Select an execution to inspect it.</p>
        )}
      </DashboardCard>

      <DashboardCard
        title={`Attempt workflow${effectiveExecution ? ` · ${effectiveExecution.executionId}` : ""}`}
        className="min-h-0 flex-1"
        bodyClassName="flex min-h-0 flex-1 flex-col gap-2 p-3"
        action={
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
            {workflowIr && (
              <span>
                {workflowIr.task_configs.length} tasks · {workflowIr.links.length} deps
              </span>
            )}
            {onViewLogs && (
              <button
                type="button"
                className="inline-flex items-center gap-1 underline-offset-2 hover:text-foreground hover:underline"
                onClick={onViewLogs}
              >
                Logs <ArrowRight className="h-3 w-3" />
              </button>
            )}
            {workflow && onOpenWorkflow && (
              <button
                type="button"
                className="inline-flex items-center gap-1 underline-offset-2 hover:text-foreground hover:underline"
                onClick={onOpenWorkflow}
              >
                Definition <ArrowRight className="h-3 w-3" />
              </button>
            )}
          </div>
        }
      >
        {workflowIr ? (
          <>
            <WorkflowGraph
              ir={workflowIr}
              height={420}
              onNodeClick={(taskId) => onInspectTask(taskId, run.id)}
            />
            {Object.keys(edgeStatusSummary).length > 0 && (
              <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                {Object.entries(edgeStatusSummary).map(([status, count]) => (
                  <span key={status} className="rounded border border-border/60 px-1.5 py-0.5">
                    {status}: {count}
                  </span>
                ))}
              </div>
            )}
            {executionGraphError && (
              <p className="text-xs text-destructive">{executionGraphError}</p>
            )}
          </>
        ) : (
          <p className="text-xs italic text-muted-foreground">
            No workflow snapshot recorded for this run.
          </p>
        )}
      </DashboardCard>
    </div>
  );
};
