import * as Popover from "@radix-ui/react-popover";
import {
  Ban,
  BarChart3,
  Check,
  Copy,
  ExternalLink,
  FileQuestion,
  FlaskConical,
  ListChecks,
  SlidersHorizontal,
  Terminal,
  Trash2,
  Workflow as WorkflowIcon,
} from "lucide-react";
import { useMemo, useState } from "react";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import type { DataTableColumn, DataTableRowAction } from "@/app/components/entity";
import {
  DashboardCard,
  DashboardGrid,
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityPage,
  StatusIcon,
} from "@/app/components/entity";
import { PlanComposer } from "@/app/components/PlanComposer";
import { countRunStatuses, formatDuration, formatScalar } from "@/app/renderers/dashboardData";
import { ExperimentCompare } from "@/app/renderers/ExperimentCompare";
import { buildExperimentWorkbenchData } from "@/app/renderers/entityWorkbenchData";
import { WorkflowGraphViewer } from "@/app/renderers/WorkflowGraphViewer";
import { MultiRunMetricsView } from "@/app/runs/metrics/MultiRunMetricsView";
import { STATUS_GROUPS } from "@/app/runs/statusGroups";
import { useRunMultiSelect } from "@/app/runs/useRunMultiSelect";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ObjectView, RendererProps, RunSummary } from "@/app/types";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { parseWorkflowIr, WorkflowGraph } from "@/components/workflow/workflow-graph";

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

const StatusDistributionBar = ({
  counts,
}: {
  counts: ReturnType<typeof countRunStatuses>;
}): JSX.Element => {
  if (counts.total === 0) return <div className="h-1.5 rounded-full bg-muted" />;
  return (
    <div className="flex h-1.5 overflow-hidden rounded-full bg-muted">
      {STATUS_GROUPS.map((group) => {
        const value = counts[group.id];
        if (value === 0) return null;
        return (
          <div
            key={group.id}
            title={`${group.label}: ${value}`}
            style={{ width: `${(value / counts.total) * 100}%`, backgroundColor: group.color }}
          />
        );
      })}
    </div>
  );
};

const ParametersCell = ({ run, keys }: { run: RunSummary; keys: string[] }): JSX.Element => {
  const entries = keys
    .map((key) => [key, run.parameters?.[key]] as const)
    .filter(([, value]) => value !== undefined);
  if (entries.length === 0) return <span className="text-xs text-muted-foreground">-</span>;
  const visible = entries.slice(0, 3);
  return (
    <div className="flex max-w-[320px] flex-wrap items-center gap-1">
      {visible.map(([key, value]) => (
        <span
          key={key}
          className="inline-flex max-w-[120px] items-center gap-1 rounded border border-border/60 bg-muted/30 px-1.5 py-0.5 text-[11px]"
          title={`${key}=${formatScalar(value)}`}
        >
          <span className="truncate text-muted-foreground">{key}</span>
          <span className="truncate font-mono text-foreground">{formatScalar(value)}</span>
        </span>
      ))}
      {entries.length > visible.length && (
        <Popover.Root>
          <Popover.Trigger asChild>
            <Button variant="ghost" size="sm" className="h-6 px-1.5 text-[11px]">
              +{entries.length - visible.length}
            </Button>
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Content
              side="bottom"
              align="start"
              className="z-50 max-h-80 w-72 overflow-auto rounded-md border border-border bg-popover p-3 text-popover-foreground shadow-md"
            >
              <dl className="space-y-2">
                {entries.map(([key, value]) => (
                  <div key={key} className="grid grid-cols-[90px_minmax(0,1fr)] gap-2 text-xs">
                    <dt className="truncate text-muted-foreground">{key}</dt>
                    <dd className="truncate font-mono text-foreground" title={formatScalar(value)}>
                      {formatScalar(value)}
                    </dd>
                  </div>
                ))}
              </dl>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
      )}
    </div>
  );
};

export const ExperimentViewer = ({
  selection,
  snapshot,
  inspectorTarget,
  onInspectorTargetChange,
  onRefresh,
}: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const { setSelection } = useNavigationState(snapshot);

  const experimentId = selection.objectId;
  const experiment = snapshot.experiments.find((e) => e.id === experimentId);
  const projectId = experiment?.projectId || "";

  const runs = useMemo(
    () => snapshot.runs.filter((r) => r.experimentId === experimentId),
    [snapshot.runs, experimentId],
  );

  const counts = useMemo(() => countRunStatuses(runs), [runs]);

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

  // Ephemeral multi-run selection (local React state, not the Zustand store) for
  // the metrics-aggregation flow: pick runs in this tab, aggregate in the next.
  const orderedRunIds = useMemo(() => runs.map((run) => run.id), [runs]);
  const runIndex = useMemo(
    () => new Map(orderedRunIds.map((id, index) => [id, index] as const)),
    [orderedRunIds],
  );
  const multi = useRunMultiSelect(orderedRunIds);
  const selectedRunIds = useMemo(
    () => runs.filter((run) => multi.selected.has(run.id)).map((run) => run.id),
    [runs, multi.selected],
  );

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

  const workflow = snapshot.workflows.find((item) => item.experimentId === experiment.id);
  const project = snapshot.projects.find((item) => item.id === projectId);
  const workflowGraph = workflow?.graph ?? parseWorkflowIr(experiment.workflowSource);
  const workbench = buildExperimentWorkbenchData(
    experiment,
    runs,
    workflowGraph ? { graph: workflowGraph } : workflow,
  );

  const runColumns: DataTableColumn<RunSummary>[] = [
    {
      key: "id",
      header: "Run",
      width: "w-[180px]",
      cell: (run) => (
        <div className="min-w-0">
          <div className="truncate text-sm font-medium text-foreground">{run.name || run.id}</div>
          <div className="truncate font-mono text-[11px] text-muted-foreground">
            {run.id.substring(0, 12)}
          </div>
        </div>
      ),
    },
    {
      key: "status",
      header: "State",
      width: "w-[70px]",
      cell: (run) => <StatusIcon status={run.status} />,
    },
    {
      key: "parameters",
      header: "Parameters",
      width: "w-[360px]",
      cell: (run) => <ParametersCell run={run} keys={parameterKeys} />,
    },
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

  // Leading tick column, shown only in multi-select mode. The cell button reads
  // the native event so shift (range) / ctrl|meta (toggle) modifiers reach the
  // pure selection reducer — DataTable's onRowClick alone carries no event.
  const selectionColumn: DataTableColumn<RunSummary> = {
    key: "select",
    header: "",
    width: "w-[36px]",
    cell: (run) => {
      const checked = multi.selected.has(run.id);
      return (
        <button
          type="button"
          aria-pressed={checked}
          aria-label={checked ? "Deselect run" : "Select run"}
          onClick={(event) => {
            event.stopPropagation();
            multi.selectAt(runIndex.get(run.id) ?? 0, {
              shift: event.shiftKey,
              meta: event.metaKey || event.ctrlKey,
            });
          }}
          className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${
            checked
              ? "border-primary bg-primary text-primary-foreground"
              : "border-border hover:border-primary"
          }`}
        >
          {checked && <Check className="h-3 w-3" />}
        </button>
      );
    },
  };
  const tableColumns = multi.enabled ? [selectionColumn, ...runColumns] : runColumns;

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
      <DashboardCard title="Generate plan with AI" className="lg:col-span-12">
        <PlanComposer
          projectId={projectId}
          experimentId={experimentId}
          onPlanComplete={onRefresh}
        />
      </DashboardCard>
      <DashboardCard title="Experiment details" className="lg:col-span-8">
        <dl className="grid gap-x-6 gap-y-3 md:grid-cols-2">
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Experiment ID
            </dt>
            <dd className="mt-0.5 truncate font-mono text-xs text-foreground">{experiment.id}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Project
            </dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {project?.name ?? projectId}
            </dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Updated
            </dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {new Date(experiment.updatedAt).toLocaleString()}
            </dd>
          </div>
        </dl>
      </DashboardCard>

      <DashboardCard title="Summary" className="lg:col-span-4" bodyClassName="space-y-3">
        <div className="flex items-baseline justify-between gap-3 border-b border-border/60 pb-2">
          <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Total runs
          </span>
          <span className="text-2xl font-semibold tabular-nums text-foreground">
            {counts.total}
          </span>
        </div>
        <StatusDistributionBar counts={counts} />
        <dl className="space-y-1.5 text-xs">
          {STATUS_GROUPS.map((group) => (
            <div key={group.id} className="flex items-center justify-between gap-3">
              <dt className="inline-flex min-w-0 items-center gap-2 text-muted-foreground">
                <span
                  className="h-1.5 w-1.5 flex-none rounded-full"
                  style={{ backgroundColor: group.color }}
                />
                <span className="truncate">{group.label}</span>
              </dt>
              <dd className="font-semibold tabular-nums text-foreground">{counts[group.id]}</dd>
            </div>
          ))}
        </dl>
      </DashboardCard>

      <DashboardCard title="Workflow" className="lg:col-span-6" bodyClassName="space-y-3">
        {workflowGraph ? (
          <>
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
              <span className="inline-flex items-center gap-1.5">
                <WorkflowIcon className="h-3.5 w-3.5 text-sky-500" />
                {workbench.workflowSummary.taskCount} tasks
              </span>
              <span>{workbench.workflowSummary.linkCount} links</span>
              <span>{workbench.workflowSummary.parallelGroupCount} parallel groups</span>
            </div>
            <WorkflowGraph ir={workflowGraph} height={230} />
          </>
        ) : (
          <p className="text-xs italic text-muted-foreground">No workflow graph recorded.</p>
        )}
      </DashboardCard>

      <DashboardCard title="Parameter space" className="lg:col-span-6" bodyClassName="space-y-3">
        {workbench.parameterAxes.length === 0 ? (
          <p className="text-xs italic text-muted-foreground">No parameter axes declared.</p>
        ) : (
          <Accordion type="multiple" className="rounded-md border border-border/60">
            {workbench.parameterAxes.map((axis) => (
              <AccordionItem key={axis.key} value={axis.key} className="border-border/60 px-2">
                <AccordionTrigger className="py-2 text-xs hover:no-underline">
                  <span className="inline-flex min-w-0 items-center gap-1.5">
                    <SlidersHorizontal className="h-3.5 w-3.5 flex-none text-muted-foreground" />
                    <span className="truncate font-medium text-foreground">{axis.key}</span>
                  </span>
                  <span className="ml-auto flex-none pr-2 tabular-nums text-muted-foreground">
                    {axis.count}
                  </span>
                </AccordionTrigger>
                <AccordionContent className="pb-2">
                  <div className="max-h-44 overflow-auto rounded-sm bg-muted/20 p-2">
                    <div className="flex flex-wrap gap-1">
                      {axis.values.map((value) => (
                        <span
                          key={`${axis.key}:${value}`}
                          className="max-w-[180px] truncate rounded border border-border/60 bg-background px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground"
                          title={value}
                        >
                          {value}
                        </span>
                      ))}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </DashboardCard>
    </DashboardGrid>
  );

  const workflowSelection = workflow
    ? { objectType: "workflow" as const, objectId: workflow.id, workflowId: workflow.id }
    : null;
  const workflowTabContent = workflowSelection ? (
    <WorkflowGraphViewer
      selection={workflowSelection}
      snapshot={snapshot}
      inspectorTarget={inspectorTarget}
      onInspectorTargetChange={onInspectorTargetChange}
      onRefresh={onRefresh}
    />
  ) : (
    <div className="flex h-full items-center justify-center">
      <EmptyState
        icon={<WorkflowIcon className="h-6 w-6" />}
        title="No workflow graph"
        description="This experiment has no linked workflow document in the snapshot."
      />
    </div>
  );

  return (
    <EntityPage
      icon={FlaskConical}
      title={experiment.name}
      status={experiment.status}
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
      activeTab={activeTab}
      onActiveTabChange={setActiveTab}
      tabs={[
        {
          value: "overview",
          label: "Overview",
          content: activeTab === "overview" ? overviewContent : null,
        },
        {
          value: "runs",
          label: `Runs${counts.total ? ` (${counts.total})` : ""}`,
          content: (
            <div className="flex h-full flex-col">
              <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-1.5">
                <Button
                  variant={multi.enabled ? "default" : "outline"}
                  size="sm"
                  className="h-7 gap-1.5"
                  aria-pressed={multi.enabled}
                  onClick={multi.toggleMode}
                  title="Select multiple runs to aggregate their metrics"
                >
                  <ListChecks className="h-3.5 w-3.5" />
                  {multi.enabled ? "Selecting" : "Select"}
                </Button>
                {multi.enabled && (
                  <>
                    <span className="text-xs text-muted-foreground">
                      {multi.selected.size} selected
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 gap-1.5"
                      disabled={multi.selected.size === 0}
                      onClick={() => setActiveTab("aggregate")}
                    >
                      <BarChart3 className="h-3.5 w-3.5" />
                      Aggregate
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7"
                      disabled={multi.selected.size === 0}
                      onClick={multi.clear}
                    >
                      Clear
                    </Button>
                    <span className="text-[11px] text-muted-foreground">
                      shift = range · ctrl/⌘ = toggle
                    </span>
                  </>
                )}
              </div>
              <DataTable
                columns={tableColumns}
                data={runs}
                getRowKey={(run) => run.id}
                onRowClick={
                  multi.enabled
                    ? (run) =>
                        multi.selectAt(runIndex.get(run.id) ?? 0, { shift: false, meta: true })
                    : (run) => navigateToRun(run.id)
                }
                rowActions={runRowActions}
                rowClassName={(run) =>
                  multi.enabled && multi.selected.has(run.id) ? "bg-primary/5" : ""
                }
                empty={
                  <EmptyState
                    title={EMPTY_COPY.runs.title}
                    description={EMPTY_COPY.runs.description}
                  />
                }
              />
            </div>
          ),
        },
        {
          value: "workflow",
          label: "Workflow",
          content: activeTab === "workflow" ? workflowTabContent : null,
        },
        {
          value: "compare",
          label: "Compare",
          content:
            activeTab === "compare" ? (
              <ExperimentCompare runs={runs} onOpenRun={navigateToRun} />
            ) : null,
        },
        {
          value: "aggregate",
          label: "Aggregate",
          content:
            activeTab === "aggregate" ? (
              <MultiRunMetricsView
                projectId={projectId}
                experimentId={experimentId}
                runIds={selectedRunIds}
              />
            ) : null,
        },
      ]}
    />
  );
};
