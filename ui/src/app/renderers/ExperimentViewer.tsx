import {
  Archive,
  Ban,
  ChevronRight,
  Copy,
  ExternalLink,
  FileQuestion,
  FlaskConical,
  Terminal,
  Trash2,
  Workflow as WorkflowIcon,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import type { DataTableColumn, DataTableRowAction } from "@/app/components/entity";
import {
  DashboardCard,
  DashboardGrid,
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityMetric,
  EntityPage,
  StatCard,
  StatGrid,
  StatusBadge,
  StatusDonut,
} from "@/app/components/entity";
import {
  countRunStatuses,
  formatDuration,
  formatScalar,
  statusDonutSegments,
  successRate,
} from "@/app/renderers/dashboardData";
import { ExperimentCompare } from "@/app/renderers/ExperimentCompare";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ObjectView, RendererProps, RunSummary } from "@/app/types";
import { Button } from "@/components/ui/button";

const formatResultPreview = (results: Record<string, unknown>): string => {
  const entries = Object.entries(results);
  if (entries.length === 0) return "—";
  if (entries.length === 1) {
    const [k, v] = entries[0];
    return `${k} = ${formatScalar(v)}`;
  }
  const head = entries
    .slice(0, 2)
    .map(([k, v]) => `${k}=${formatScalar(v)}`)
    .join(", ");
  return entries.length > 2 ? `${head}, +${entries.length - 2}` : head;
};

interface WorkflowPreview {
  taskNames: string[];
  edgeCount: number;
}

const asRecord = (value: unknown): Record<string, unknown> | null => {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
};

const extractTaskNames = (root: Record<string, unknown>): string[] => {
  const graph = asRecord(root.graph);
  if (graph && Array.isArray(graph.nodes)) {
    return graph.nodes
      .map((node) => {
        const rec = asRecord(node);
        if (!rec) return null;
        return typeof rec.label === "string"
          ? rec.label
          : typeof rec.id === "string"
            ? rec.id
            : null;
      })
      .filter((v): v is string => Boolean(v));
  }
  const tasks = Array.isArray(root.tasks) ? root.tasks : null;
  if (tasks) {
    return tasks
      .map((task) => {
        const rec = asRecord(task);
        return typeof rec?.id === "string" ? rec.id : null;
      })
      .filter((v): v is string => Boolean(v));
  }
  return [];
};

const fetchWorkflowPreview = async (path: string): Promise<WorkflowPreview | null> => {
  if (
    !path ||
    path.includes(":") ||
    (!path.endsWith(".json") && !path.endsWith(".yaml") && !path.endsWith(".yml"))
  ) {
    return null;
  }
  try {
    const response = await fetch(`/api/workspace/files?path=${encodeURIComponent(path)}`);
    if (!response.ok) return null;
    const data = (await response.json()) as unknown;
    const root = asRecord(data);
    if (!root) return null;
    const inner = asRecord(asRecord(root.context)?.workflow) ?? root;
    const taskNames = extractTaskNames(inner);
    const graph = asRecord(inner.graph);
    const edges = graph && Array.isArray(graph.edges) ? graph.edges.length : 0;
    if (taskNames.length === 0 && edges === 0) return null;
    return { taskNames, edgeCount: edges };
  } catch {
    return null;
  }
};

export const ExperimentViewer = ({
  selection,
  snapshot,
  onRefresh,
}: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [workflowPreview, setWorkflowPreview] = useState<WorkflowPreview | null>(null);
  const { setSelection } = useNavigationState(snapshot);

  const experimentId = selection.objectId;
  const experiment = snapshot.experiments.find((e) => e.id === experimentId);
  const projectId = experiment?.projectId || "";

  const workflowPath = experiment?.workflowFile || experiment?.workflowSource || null;
  useEffect(() => {
    let cancelled = false;
    if (!workflowPath) {
      setWorkflowPreview(null);
      return;
    }
    fetchWorkflowPreview(workflowPath).then((preview) => {
      if (!cancelled) setWorkflowPreview(preview);
    });
    return () => {
      cancelled = true;
    };
  }, [workflowPath]);

  const runs = useMemo(
    () => snapshot.runs.filter((r) => r.experimentId === experimentId),
    [snapshot.runs, experimentId],
  );

  const counts = useMemo(() => countRunStatuses(runs), [runs]);
  const donut = useMemo(() => statusDonutSegments(counts), [counts]);
  const rate = successRate(counts);

  const recentRuns = useMemo(() => {
    return [...runs]
      .sort((a, b) => {
        const aT = Date.parse(a.finishedAt ?? a.startedAt ?? a.updatedAt ?? "") || 0;
        const bT = Date.parse(b.finishedAt ?? b.startedAt ?? b.updatedAt ?? "") || 0;
        return bT - aT;
      })
      .slice(0, 6);
  }, [runs]);

  // Union of parameter keys across all runs — stable first-seen order. Declared
  // before any early return so the hook order is unconditional.
  const parameterKeys = useMemo(() => {
    const seen = new Set<string>();
    const order: string[] = [];
    for (const run of runs) {
      for (const key of Object.keys(run.parameters ?? {})) {
        if (!seen.has(key)) {
          seen.add(key);
          order.push(key);
        }
      }
    }
    return order;
  }, [runs]);

  const handleDelete = async () => {
    if (!projectId) return;
    if (!confirm(`Are you sure you want to delete experiment "${experimentId}"?`)) {
      return;
    }
    setIsDeleting(true);
    try {
      await workspaceApi.deleteExperiment(projectId, experimentId);
      onRefresh();
    } catch (error) {
      console.error("Failed to delete experiment:", error);
      alert("Failed to delete experiment");
    } finally {
      setIsDeleting(false);
    }
  };

  const navigateToRun = (runId: string) => {
    setSelection({ objectType: "run", objectId: runId });
  };

  const navigateToRunView = (run: RunSummary, objectView?: ObjectView) => {
    setSelection({ objectType: "run", objectId: run.id, objectView });
  };

  const handleCancelRun = async (run: RunSummary) => {
    if (["succeeded", "failed", "cancelled", "skipped"].includes(run.status)) {
      return;
    }
    if (
      !confirm(
        `Mark run "${run.id}" as cancelled?\n\nThis updates workspace status only; it does not cancel a scheduler job.`,
      )
    ) {
      return;
    }
    try {
      await workspaceApi.updateRunStatus(run.projectId, run.experimentId, run.id, "cancelled");
      onRefresh();
    } catch (error) {
      console.error("Failed to mark run cancelled:", error);
      alert("Failed to mark run cancelled");
    }
  };

  const copyToClipboard = (text: string) => {
    void navigator.clipboard.writeText(text);
  };

  if (!experiment || !projectId) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<FileQuestion className="h-6 w-6" />}
          title="Experiment not found"
          description="It may have been deleted or not yet synced."
        />
      </div>
    );
  }

  const project = snapshot.projects.find((item) => item.id === projectId);
  const workflow = snapshot.workflows.find((item) => item.experimentId === experiment.id);
  const parameterAxes = Object.entries(experiment.parameterSpace ?? {});

  const parameterColumns: DataTableColumn<RunSummary>[] = parameterKeys.map((key) => ({
    key: `param:${key}`,
    header: key,
    width: "w-[110px]",
    cell: (run) => {
      const value = run.parameters?.[key];
      return (
        <span className="font-mono text-xs text-foreground">
          {value === undefined ? (
            <span className="text-muted-foreground">—</span>
          ) : (
            formatScalar(value)
          )}
        </span>
      );
    },
  }));

  const runColumns: DataTableColumn<RunSummary>[] = [
    {
      key: "id",
      header: "Run",
      width: "w-[110px]",
      cell: (run) => (
        <span className="font-mono text-xs text-muted-foreground">{run.id.substring(0, 8)}</span>
      ),
    },
    {
      key: "status",
      header: "Status",
      width: "w-[120px]",
      cell: (run) => <StatusBadge status={run.status} />,
    },
    ...parameterColumns,
    {
      key: "result",
      header: "Result",
      cell: (run) => {
        const preview = formatResultPreview(run.results ?? {});
        const full = JSON.stringify(run.results ?? {}, null, 2);
        return (
          <span
            className="block max-w-[260px] truncate font-mono text-xs text-foreground"
            title={full}
          >
            {preview}
          </span>
        );
      },
    },
    {
      key: "duration",
      header: "Duration",
      width: "w-[100px]",
      cell: (run) => {
        const d = formatDuration(run.startedAt, run.finishedAt);
        return (
          <span className="font-mono text-xs text-muted-foreground">
            {d ?? <span className="text-muted-foreground">—</span>}
          </span>
        );
      },
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-[160px]",
      cell: (run) => (
        <span className="text-xs text-muted-foreground">
          {new Date(run.updatedAt).toLocaleString()}
        </span>
      ),
    },
    {
      key: "action",
      header: "",
      width: "w-[40px]",
      align: "right",
      cell: (run) => (
        <Button
          size="icon"
          variant="ghost"
          aria-label="Open run"
          className="h-6 w-6 text-muted-foreground opacity-60 transition-opacity group-hover:opacity-100 hover:text-foreground"
          onClick={(event) => {
            event.stopPropagation();
            navigateToRunView(run);
          }}
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </Button>
      ),
    },
  ];

  const runRowActions = (run: RunSummary): DataTableRowAction<RunSummary>[] => [
    {
      id: "open",
      label: "Open run",
      icon: ExternalLink,
      onSelect: () => navigateToRunView(run),
    },
    {
      id: "logs",
      label: "View logs",
      icon: Terminal,
      onSelect: () => navigateToRunView(run, "logs"),
    },
    {
      id: "snapshot",
      label: "View snapshot",
      icon: Archive,
      onSelect: () => navigateToRunView(run, "snapshot"),
    },
    {
      id: "copy-id",
      label: "Copy run ID",
      icon: Copy,
      onSelect: () => copyToClipboard(run.id),
    },
    {
      id: "cancel",
      label: "Mark cancelled",
      icon: Ban,
      disabled: ["succeeded", "failed", "cancelled", "skipped"].includes(run.status),
      destructive: true,
      separatorBefore: true,
      title: "Updates workspace status only; it does not cancel a scheduler job.",
      onSelect: () => {
        void handleCancelRun(run);
      },
    },
  ];

  const overviewContent = (
    <DashboardGrid>
      {/* KPI band */}
      <div className="lg:col-span-12">
        <StatGrid>
          <StatCard label="Runs" value={counts.total} muted={counts.total === 0} />
          <StatCard
            label="Succeeded"
            value={counts.succeeded}
            tone="success"
            muted={counts.succeeded === 0}
          />
          <StatCard label="Failed" value={counts.failed} tone="error" muted={counts.failed === 0} />
          <StatCard
            label="Running"
            value={counts.running}
            tone="running"
            muted={counts.running === 0}
          />
          <StatCard
            label="Success rate"
            value={rate === null ? "—" : `${rate.toFixed(0)}%`}
            hint={
              rate === null
                ? "no terminal runs"
                : `${counts.succeeded}/${counts.succeeded + counts.failed + counts.cancelled} terminal`
            }
            tone={
              rate === null ? "neutral" : rate >= 80 ? "success" : rate >= 50 ? "warning" : "error"
            }
            muted={rate === null}
          />
        </StatGrid>
      </div>

      {/* Status mix donut */}
      <DashboardCard title="Status mix" className="lg:col-span-5">
        {counts.total === 0 ? (
          <p className="text-xs italic text-muted-foreground">No runs launched yet.</p>
        ) : (
          <StatusDonut segments={donut} centerLabel="runs" />
        )}
      </DashboardCard>

      {/* Recent runs */}
      <DashboardCard
        title="Recent runs"
        className="lg:col-span-7"
        bodyClassName="p-0"
        action={
          counts.total > 0 ? (
            <span className="text-[11px] tabular-nums text-muted-foreground">
              {Math.min(recentRuns.length, counts.total)} of {counts.total}
            </span>
          ) : undefined
        }
      >
        {recentRuns.length === 0 ? (
          <p className="p-3 text-xs italic text-muted-foreground">No runs yet.</p>
        ) : (
          <ul className="divide-y divide-border/50">
            {recentRuns.map((run) => {
              const duration = formatDuration(run.startedAt, run.finishedAt);
              const when = run.finishedAt ?? run.startedAt ?? run.updatedAt;
              return (
                <li key={run.id}>
                  <button
                    type="button"
                    onClick={() => navigateToRunView(run)}
                    className="grid w-full grid-cols-[auto_minmax(0,1fr)_auto_auto] items-center gap-3 px-3 py-2 text-left text-xs transition-colors hover:bg-muted/40"
                  >
                    <StatusBadge status={run.status} size="sm" />
                    <span className="min-w-0 truncate font-medium text-foreground">
                      {run.name || run.id.substring(0, 8)}
                    </span>
                    <span className="font-mono text-muted-foreground">{duration ?? "—"}</span>
                    <span className="hidden text-muted-foreground sm:inline">
                      {when ? new Date(when).toLocaleString() : "—"}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </DashboardCard>

      {/* Parameter space */}
      <DashboardCard title="Parameter space" className="lg:col-span-6">
        {parameterAxes.length === 0 ? (
          <p className="text-xs italic text-muted-foreground">No parameter axes declared.</p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {parameterAxes.map(([key, value]) => (
              <span
                key={key}
                className="inline-flex items-center gap-1.5 rounded-md border border-border/60 bg-muted/30 px-2 py-1 text-xs"
                title={formatScalar(value)}
              >
                <span className="font-medium text-foreground">{key}</span>
                <span className="max-w-[160px] truncate font-mono text-[11px] text-muted-foreground">
                  {formatScalar(value)}
                </span>
              </span>
            ))}
          </div>
        )}
      </DashboardCard>

      {/* Workflow */}
      <DashboardCard
        title="Workflow"
        className="lg:col-span-6"
        action={
          workflow && (experiment.workflowSource || experiment.workflowFile) ? (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 gap-1 px-1.5 text-[11px] text-muted-foreground"
              onClick={() =>
                setSelection({
                  objectType: "workflow",
                  objectId: workflow.id,
                  workflowId: workflow.id,
                })
              }
            >
              Open <ChevronRight className="h-3 w-3" />
            </Button>
          ) : undefined
        }
      >
        {workflow && (experiment.workflowSource || experiment.workflowFile) ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <WorkflowIcon className="h-4 w-4 flex-none text-sky-500" />
              <span className="min-w-0 break-all font-mono text-xs text-foreground">
                {experiment.workflowSource || experiment.workflowFile}
              </span>
            </div>
            {workflowPreview ? (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                  {workflowPreview.taskNames.length} tasks
                  {workflowPreview.edgeCount > 0 ? ` · ${workflowPreview.edgeCount} edges` : ""}
                </span>
                {workflowPreview.taskNames.slice(0, 8).map((name) => (
                  <span
                    key={name}
                    className="rounded-sm border border-border/60 bg-background px-1.5 py-0.5 font-mono text-[11px] text-foreground"
                  >
                    {name}
                  </span>
                ))}
                {workflowPreview.taskNames.length > 8 && (
                  <span className="text-[11px] text-muted-foreground">
                    +{workflowPreview.taskNames.length - 8}
                  </span>
                )}
              </div>
            ) : (
              <p className="text-[11px] text-muted-foreground">Open to inspect the task graph.</p>
            )}
          </div>
        ) : (
          <p className="text-xs italic text-muted-foreground">No workflow file recorded.</p>
        )}
      </DashboardCard>

      {/* Details */}
      <DashboardCard title="Details" className="lg:col-span-12">
        <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="min-w-0">
            <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">Project</dt>
            <dd className="mt-0.5">
              <button
                type="button"
                onClick={() => setSelection({ objectType: "project", objectId: projectId })}
                className="truncate text-sm text-foreground hover:text-primary hover:underline"
              >
                {project?.name || projectId}
              </button>
            </dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">
              Experiment ID
            </dt>
            <dd className="mt-0.5 truncate font-mono text-sm text-foreground">{experiment.id}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">
              Workflow file
            </dt>
            <dd className="mt-0.5 truncate font-mono text-sm text-foreground">
              {experiment.workflowFile || "—"}
            </dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">Updated</dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {new Date(experiment.updatedAt).toLocaleString()}
            </dd>
          </div>
        </dl>
      </DashboardCard>
    </DashboardGrid>
  );

  return (
    <EntityPage
      icon={FlaskConical}
      title={experiment.name}
      status={experiment.status}
      subtitle={experiment.summary || undefined}
      actions={
        <>
          <CreateRunDialog
            projectId={projectId}
            experimentId={experimentId}
            workflowFile={experiment.workflowFile || ""}
            onRunCreated={onRefresh}
          />
          <Button
            variant="ghost"
            size="icon"
            onClick={handleDelete}
            disabled={isDeleting}
            className="h-7 w-7 text-muted-foreground hover:text-destructive"
            aria-label="Delete experiment"
            title="Delete experiment"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </>
      }
      metrics={
        <>
          <EntityMetric label="Runs" value={counts.total} />
          <EntityMetric label="OK" value={counts.succeeded} />
          <EntityMetric label="Failed" value={counts.failed} />
          <EntityMetric label="Running" value={counts.running} />
        </>
      }
      tabs={[
        {
          value: "overview",
          label: "Overview",
          content: overviewContent,
        },
        {
          value: "runs",
          label: `Runs${counts.total ? ` (${counts.total})` : ""}`,
          content: (
            <DataTable
              columns={runColumns}
              data={runs}
              getRowKey={(run) => run.id}
              onRowClick={(run) => navigateToRun(run.id)}
              rowActions={runRowActions}
              empty={
                <EmptyState
                  title={EMPTY_COPY.runs.title}
                  description={EMPTY_COPY.runs.description}
                />
              }
            />
          ),
        },
        {
          value: "compare",
          label: "Compare",
          content: <ExperimentCompare runs={runs} onOpenRun={navigateToRun} />,
        },
      ]}
    />
  );
};
