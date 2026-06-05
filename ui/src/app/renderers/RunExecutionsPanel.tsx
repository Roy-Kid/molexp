import { ArrowRight, GitBranch } from "lucide-react";
import type { JSX } from "react";
import { DashboardCard, EmptyState, StatusBadge } from "@/app/components/entity";
import { formatDuration } from "@/app/renderers/dashboardData";
import type { RunSummary, WorkflowSummary } from "@/app/types";
import { parseWorkflowIr, WorkflowGraph } from "@/components/workflow/workflow-graph";

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
  const workflowIr = parseWorkflowIr(run.workflowSource);

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

  const effectiveExecutionId =
    selectedExecutionId ?? history[history.length - 1]?.executionId ?? null;
  const effectiveExecution = history.find((rec) => rec.executionId === effectiveExecutionId);

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-auto p-4">
      <ol className="flex flex-wrap gap-2">
        {history.map((rec, index) => {
          const active = rec.executionId === effectiveExecutionId;
          const d = formatDuration(rec.startedAt, rec.finishedAt);
          return (
            <li key={rec.executionId}>
              <button
                type="button"
                onClick={() => onSelectExecution(rec.executionId)}
                className={`flex min-w-[150px] flex-col gap-1 rounded-lg border px-3 py-2 text-left text-xs transition-colors ${
                  active
                    ? "border-foreground/30 bg-muted/50 ring-1 ring-inset ring-foreground/20"
                    : "border-border/60 bg-card hover:bg-muted/30"
                }`}
                title={rec.executionId}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-muted-foreground">#{index + 1}</span>
                  <StatusBadge status={rec.status} size="sm" />
                </div>
                <span className="truncate font-mono text-[11px] text-foreground">
                  {rec.executionId}
                </span>
                <span className="text-[11px] text-muted-foreground">
                  {formatTimeOfDay(rec.startedAt)}
                  {d ? ` · ${d}` : ""}
                </span>
              </button>
            </li>
          );
        })}
      </ol>

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
          <WorkflowGraph
            ir={workflowIr}
            height={420}
            onNodeClick={(taskId) => onInspectTask(taskId, run.id)}
          />
        ) : (
          <p className="text-xs italic text-muted-foreground">
            No workflow snapshot recorded for this run.
          </p>
        )}
      </DashboardCard>
    </div>
  );
};
