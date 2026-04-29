import {
  Archive,
  Ban,
  ChevronRight,
  Code2,
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
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityMetric,
  EntityPage,
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
  StatusBadge,
} from "@/app/components/entity";
import { SnapshotDiffPanel } from "@/app/renderers/SnapshotViewer";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ObjectView, RendererProps, RunSummary } from "@/app/types";
import { Button } from "@/components/ui/button";

const formatScalar = (value: unknown): string => {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const formatResultPreview = (results: Record<string, unknown>): string => {
  const entries = Object.entries(results);
  if (entries.length === 0) return "—";
  if (entries.length === 1) {
    const [k, v] = entries[0];
    return `${k} = ${formatScalar(v)}`;
  }
  // Up to 2 keys inline; rest collapsed.
  const head = entries
    .slice(0, 2)
    .map(([k, v]) => `${k}=${formatScalar(v)}`)
    .join(", ");
  return entries.length > 2 ? `${head}, +${entries.length - 2}` : head;
};

const formatDuration = (startIso: string | null, endIso: string | null): string | null => {
  if (!startIso || !endIso) return null;
  const start = Date.parse(startIso);
  const end = Date.parse(endIso);
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  const ms = Math.max(0, end - start);
  if (ms < 1000) return `${ms}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds - minutes * 60);
  return `${minutes}m${remainder}s`;
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
  // Prefer graph.nodes[].label, then tasks[].id (workflow.json schema).
  const graph = asRecord(root.graph);
  if (graph && Array.isArray(graph.nodes)) {
    return graph.nodes
      .map((node) => {
        const rec = asRecord(node);
        if (!rec) return null;
        const label =
          typeof rec.label === "string" ? rec.label : typeof rec.id === "string" ? rec.id : null;
        return label;
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
  // Code references like "module.py:function" aren't readable as workflow
  // graphs — bail rather than emitting a 404 in the console.
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
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);

  // Find the experiment in snapshot
  const experimentId = selection.objectId;
  const experiment = snapshot.experiments.find((e) => e.id === experimentId);
  const projectId = experiment?.projectId || "";

  // Lazy preview of the workflow graph: parse the workflow file (yaml/json)
  // when one is bound. Function references like "module.py:fn" can't be
  // parsed as graphs and are skipped (we still show the source path + jump).
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

  // Filter runs for this experiment
  const runs = useMemo(
    () => snapshot.runs.filter((r) => r.experimentId === experimentId),
    [snapshot.runs, experimentId],
  );

  const stats = useMemo(() => {
    return {
      total: runs.length,
      succeeded: runs.filter((r) => r.status === "succeeded").length,
      failed: runs.filter((r) => r.status === "failed").length,
      running: runs.filter((r) => r.status === "running").length,
    };
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
    setSelection({
      objectType: "run",
      objectId: runId,
    });
  };

  const navigateToRunView = (run: RunSummary, objectView?: ObjectView) => {
    setSelection({
      objectType: "run",
      objectId: run.id,
      objectView,
    });
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
  // mapWorkflows synthesizes one WorkflowSummary per experiment with
  // id=`workflow:<experimentId>`. Match by experimentId so the lookup
  // is robust regardless of the workflowFile string.
  const workflow = snapshot.workflows.find((item) => item.experimentId === experiment.id);

  // Union of parameter keys across all runs in this experiment so each
  // parameter dimension gets its own column. Stable order = first-seen.
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

  const parameterColumns: DataTableColumn<RunSummary>[] = parameterKeys.map((key) => ({
    key: `param:${key}`,
    header: key,
    width: "w-[110px]",
    cell: (run) => {
      const value = (run.parameters ?? {})[key];
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

  return (
    <EntityPage
      breadcrumbs={breadcrumbs}
      canNavigateUp={canNavigateUp}
      onNavigateUp={navigateUp}
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
          <EntityMetric label="Runs" value={stats.total} />
          <EntityMetric label="Succeeded" value={stats.succeeded} />
          <EntityMetric label="Failed" value={stats.failed} />
          <EntityMetric label="Running" value={stats.running} />
        </>
      }
      tabs={[
        {
          value: "runs",
          label: "Runs",
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
          value: "overview",
          label: "Overview",
          content: (
            <OverviewPage
              aside={
                <>
                  <OverviewSection title="Run Summary">
                    <OverviewHighlightGrid>
                      <OverviewHighlight label="Total" value={stats.total} detail="runs" />
                      <OverviewHighlight label="Succeeded" value={stats.succeeded} />
                      <OverviewHighlight label="Failed" value={stats.failed} />
                      <OverviewHighlight label="Running" value={stats.running} />
                    </OverviewHighlightGrid>
                  </OverviewSection>

                  <OverviewSection title="Relationships">
                    <div className="flex flex-wrap gap-1.5">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => setSelection({ objectType: "project", objectId: projectId })}
                      >
                        Project: {project?.name || projectId}
                      </Button>
                      {workflow && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2 text-xs"
                          onClick={() =>
                            setSelection({
                              objectType: "workflow",
                              objectId: workflow.id,
                              workflowId: workflow.id,
                            })
                          }
                        >
                          Workflow: {workflow.name}
                        </Button>
                      )}
                    </div>
                  </OverviewSection>
                </>
              }
            >
              <OverviewSection title="Summary">
                <p className="max-w-3xl text-sm leading-6 text-foreground">
                  {experiment.summary || (
                    <span className="text-muted-foreground">No summary provided.</span>
                  )}
                </p>
              </OverviewSection>

              {Object.keys(experiment.parameterSpace ?? {}).length > 0 && (
                <OverviewSection
                  title="Parameter sweep"
                  description="Declared parameter axes for this experiment."
                >
                  <KeyValueGrid
                    items={Object.entries(experiment.parameterSpace).map(([key, value]) => ({
                      label: key,
                      value: <span className="font-mono text-xs">{formatScalar(value)}</span>,
                    }))}
                  />
                </OverviewSection>
              )}

              <OverviewSection title="Workflow">
                {workflow && (experiment.workflowSource || experiment.workflowFile) ? (
                  <button
                    type="button"
                    className="group flex w-full max-w-3xl items-start gap-3 rounded-md border border-border/70 bg-muted/30 p-3 text-left transition-colors hover:border-border hover:bg-muted/50 focus:outline-none focus:ring-2 focus:ring-ring"
                    onClick={() =>
                      setSelection({
                        objectType: "workflow",
                        objectId: workflow.id,
                        workflowId: workflow.id,
                      })
                    }
                  >
                    <WorkflowIcon className="mt-0.5 h-4 w-4 flex-none text-muted-foreground" />
                    <div className="min-w-0 flex-1 space-y-2">
                      <div className="break-all font-mono text-xs text-foreground">
                        {experiment.workflowSource || experiment.workflowFile}
                      </div>
                      {workflowPreview ? (
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                            {workflowPreview.taskNames.length} tasks
                            {workflowPreview.edgeCount > 0
                              ? ` · ${workflowPreview.edgeCount} edges`
                              : ""}
                          </span>
                          {workflowPreview.taskNames.slice(0, 6).map((name) => (
                            <span
                              key={name}
                              className="rounded-sm border border-border/70 bg-background px-1.5 py-0.5 font-mono text-[11px] text-foreground"
                            >
                              {name}
                            </span>
                          ))}
                          {workflowPreview.taskNames.length > 6 && (
                            <span className="text-[11px] text-muted-foreground">
                              +{workflowPreview.taskNames.length - 6} more
                            </span>
                          )}
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                          <Code2 className="h-3 w-3" />
                          <span>Click to open in the workflow viewer</span>
                        </div>
                      )}
                    </div>
                    <ChevronRight className="mt-0.5 h-4 w-4 flex-none text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-foreground" />
                  </button>
                ) : (
                  <p className="text-sm italic text-muted-foreground">No workflow file recorded.</p>
                )}
              </OverviewSection>

              <OverviewSection title="Metadata">
                <KeyValueGrid
                  items={[
                    { label: "Experiment ID", value: experiment.id },
                    { label: "Project ID", value: projectId },
                    { label: "Workflow File", value: experiment.workflowFile || "-" },
                    { label: "Updated", value: new Date(experiment.updatedAt).toLocaleString() },
                  ]}
                />
              </OverviewSection>
            </OverviewPage>
          ),
        },
        {
          value: "diff",
          label: "Diff",
          content: <SnapshotDiffPanel experimentRunIds={runs.map((r) => r.id)} />,
        },
      ]}
    />
  );
};
