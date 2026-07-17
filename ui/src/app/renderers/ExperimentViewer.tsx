import {
  BarChart3,
  Bot,
  Check,
  FileQuestion,
  FlaskConical,
  ListChecks,
  SlidersHorizontal,
  Trash2,
  Workflow as WorkflowIcon,
} from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import { CreateSweepDialog } from "@/app/components/CreateSweepDialog";
import { CurateComposer } from "@/app/components/CurateComposer";
import type { DataTableColumn, DataTableRowAction } from "@/app/components/entity";
import {
  DashboardCard,
  DashboardGrid,
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityPage,
  MetaField,
  MetaGrid,
  ParamChip,
  StatCard,
  StatGrid,
  StatusDistribution,
  StatusIcon,
} from "@/app/components/entity";
// Module-path import (not the barrel) — see RunViewer.tsx for the loader rationale.
import { KnowledgeBacklinksCard } from "@/app/components/entity/KnowledgeBacklinksCard";
import { countRunStatuses, formatDuration, formatScalar } from "@/app/renderers/dashboardData";
import { ExperimentCompare } from "@/app/renderers/ExperimentCompare";
import { buildExperimentWorkbenchData } from "@/app/renderers/entityWorkbenchData";
import { WorkflowGraphViewer } from "@/app/renderers/WorkflowGraphViewer";
import { MultiRunMetricsView } from "@/app/runs/metrics/MultiRunMetricsView";
import { canCancel } from "@/app/runs/runLifecycle";
import {
  buildRunListActions,
  primaryRunVerb,
  type RunListHandlers,
} from "@/app/runs/runListActions";
import { useRunMultiSelect } from "@/app/runs/useRunMultiSelect";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ObjectView, RendererProps, RunSummary } from "@/app/types";
import { useConfirm } from "@/components/ConfirmDialog";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { toast } from "@/components/ui/toast";
import { parseWorkflowIr, WorkflowGraph } from "@/components/workflow/workflow-graph";
import { formatDateTime } from "@/lib/datetime";

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

const ParametersCell = ({ run, keys }: { run: RunSummary; keys: string[] }): JSX.Element => {
  const entries = keys
    .map((key) => [key, run.parameters?.[key]] as const)
    .filter(([, value]) => value !== undefined);
  if (entries.length === 0) return <span className="text-xs text-muted-foreground">—</span>;
  const visible = entries.slice(0, 3);
  return (
    <div className="flex max-w-[320px] flex-wrap items-center gap-1">
      {visible.map(([key, value]) => (
        <ParamChip key={key} name={key} value={formatScalar(value)} />
      ))}
      {entries.length > visible.length && (
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="sm" className="h-6 px-1.5 text-[11px]">
              +{entries.length - visible.length}
            </Button>
          </PopoverTrigger>
          <PopoverContent side="bottom" align="start" className="max-h-80 w-72 overflow-auto p-3">
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
          </PopoverContent>
        </Popover>
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
  const { confirm, dialog: confirmDialog } = useConfirm();

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
    if (!window.confirm(`Delete “${experimentId}”?`)) {
      return;
    }
    setIsDeleting(true);
    try {
      await workspaceApi.deleteExperiment(projectId, experimentId);
      onRefresh();
    } catch (error) {
      console.error("Failed to delete experiment:", error);
    } finally {
      setIsDeleting(false);
    }
  };

  const navigateToRun = (runId: string) => {
    setSelection({ objectType: "run", objectId: runId });
  };

  const navigateToRunView = useCallback(
    (run: RunSummary, objectView?: ObjectView) => {
      setSelection({ objectType: "run", objectId: run.id, objectView });
    },
    [setSelection],
  );

  const handleCancelRun = useCallback(
    async (run: RunSummary) => {
      if (!canCancel(run.status)) return;
      const ok = await confirm({
        title: "Cancel run?",
        description: (
          <>
            Stop <code className="rounded bg-muted px-1 py-0.5 text-xs">{run.id}</code>?
          </>
        ),
        confirmLabel: "Cancel",
        destructive: true,
      });
      if (!ok) return;
      try {
        await workspaceApi.killRun(run.projectId, run.experimentId, run.id);
        toast.success("Cancelled");
        onRefresh();
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Cancel failed");
      }
    },
    [confirm, onRefresh],
  );

  const handleResumeRun = useCallback(
    async (run: RunSummary) => {
      try {
        await workspaceApi.resumeRun(run.projectId, run.experimentId, run.id);
        toast.success("Resumed");
        onRefresh();
        navigateToRunView(run, "executions");
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Resume failed");
      }
    },
    [navigateToRunView, onRefresh],
  );

  const handleRerunRun = useCallback(
    async (run: RunSummary, fresh = false) => {
      try {
        await workspaceApi.rerunRun(run.projectId, run.experimentId, run.id, fresh);
        toast.success(fresh ? "Rerun fresh" : "Rerun");
        onRefresh();
        navigateToRunView(run, "executions");
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Rerun failed");
      }
    },
    [navigateToRunView, onRefresh],
  );

  const runListHandlers: RunListHandlers = useMemo(
    () => ({
      open: navigateToRunView,
      cancel: (run) => {
        void handleCancelRun(run);
      },
      resume: (run) => {
        void handleResumeRun(run);
      },
      rerun: (run, fresh) => {
        void handleRerunRun(run, fresh);
      },
      copyId: (run) => {
        void navigator.clipboard.writeText(run.id);
        toast.success("Copied");
      },
    }),
    [handleCancelRun, handleResumeRun, handleRerunRun, navigateToRunView],
  );

  if (!experiment || !projectId) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState icon={<FileQuestion className="h-6 w-6" />} title="Not found" />
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
        <span className="text-xs text-muted-foreground" title={run.updatedAt}>
          {formatDateTime(run.updatedAt)}
        </span>
      ),
    },
    {
      key: "action",
      header: "",
      width: "w-[88px]",
      align: "right",
      cell: (run) => {
        const verb = primaryRunVerb(run.status);
        if (!verb) return null;
        return (
          <Button
            size="sm"
            variant={verb.kind === "cancel" ? "outline" : "default"}
            className={`h-6 px-2 text-[11px] ${
              verb.kind === "cancel"
                ? "text-destructive hover:bg-destructive/10 hover:text-destructive"
                : ""
            }`}
            aria-label={verb.label}
            onClick={(event) => {
              event.stopPropagation();
              if (verb.kind === "start") navigateToRunView(run);
              else if (verb.kind === "cancel") void handleCancelRun(run);
              else if (verb.kind === "resume") void handleResumeRun(run);
            }}
          >
            {verb.label}
          </Button>
        );
      },
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

  const runRowActions = (run: RunSummary): DataTableRowAction<RunSummary>[] =>
    buildRunListActions(run, runListHandlers).map((action) => ({
      id: action.id,
      label: action.label,
      icon: action.icon,
      disabled: action.disabled,
      destructive: action.destructive,
      separatorBefore: action.separatorBefore,
      title: action.title,
      onSelect: () => action.onSelect(),
    }));

  const overviewContent = (
    <DashboardGrid>
      <div className="lg:col-span-12">
        <StatGrid>
          <StatCard label="Runs" value={counts.total} muted={counts.total === 0} />
          <StatCard
            label="Succeeded"
            value={counts.succeeded}
            tone="success"
            muted={counts.succeeded === 0}
          />
          <StatCard
            label="Running"
            value={counts.running}
            tone="running"
            muted={counts.running === 0}
          />
          <StatCard label="Failed" value={counts.failed} tone="error" muted={counts.failed === 0} />
          <StatCard
            label="Pending"
            value={counts.pending}
            tone="warning"
            muted={counts.pending === 0}
          />
        </StatGrid>
      </div>

      <DashboardCard title="Identity" description="Experiment metadata" className="lg:col-span-5">
        <MetaGrid columns={2}>
          <MetaField label="Experiment ID" value={experiment.id} mono title={experiment.id} />
          <MetaField label="Project" value={project?.name ?? projectId} />
          <MetaField
            label="Updated"
            value={formatDateTime(experiment.updatedAt)}
            title={experiment.updatedAt}
          />
          <MetaField
            label="Workflow tasks"
            value={workbench.workflowSummary.exists ? workbench.workflowSummary.taskCount : "—"}
          />
        </MetaGrid>
      </DashboardCard>

      <DashboardCard
        title="Run status"
        description={
          counts.total === 0
            ? "No runs yet"
            : `${counts.total} run${counts.total === 1 ? "" : "s"} in this experiment`
        }
        className="lg:col-span-7"
      >
        <StatusDistribution counts={counts} />
      </DashboardCard>

      <KnowledgeBacklinksCard
        kind="experiment"
        projectId={experiment.projectId}
        experimentId={experiment.id}
        className="lg:col-span-4"
      />

      <DashboardCard
        title="Workflow"
        description={
          workflowGraph
            ? `${workbench.workflowSummary.taskCount} tasks · ${workbench.workflowSummary.linkCount} links · ${workbench.workflowSummary.parallelGroupCount} parallel`
            : "No graph recorded"
        }
        className="lg:col-span-8"
        bodyClassName="space-y-3"
        action={
          workflowGraph ? (
            <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
              <WorkflowIcon className="h-3.5 w-3.5" />
              Graph
            </span>
          ) : undefined
        }
      >
        {workflowGraph ? (
          <WorkflowGraph ir={workflowGraph} height={240} />
        ) : (
          <p className="text-sm text-muted-foreground">No workflow graph recorded.</p>
        )}
      </DashboardCard>

      <DashboardCard
        title="Parameter space"
        description={
          workbench.parameterAxes.length === 0
            ? "No axes declared"
            : `${workbench.parameterAxes.length} axis${workbench.parameterAxes.length === 1 ? "" : "es"}`
        }
        className="lg:col-span-6"
        bodyClassName="space-y-3"
      >
        {workbench.parameterAxes.length === 0 ? (
          <p className="text-sm text-muted-foreground">No parameter axes declared.</p>
        ) : (
          <Accordion type="multiple" className="rounded-lg border border-border">
            {workbench.parameterAxes.map((axis) => (
              <AccordionItem key={axis.key} value={axis.key} className="border-border px-3">
                <AccordionTrigger className="py-2.5 text-xs hover:no-underline">
                  <span className="inline-flex min-w-0 items-center gap-1.5">
                    <SlidersHorizontal className="h-3.5 w-3.5 flex-none text-muted-foreground" />
                    <span className="truncate font-medium text-foreground">{axis.key}</span>
                  </span>
                  <Badge variant="secondary" className="ml-auto mr-2 font-mono text-[10px]">
                    {axis.count}
                  </Badge>
                </AccordionTrigger>
                <AccordionContent className="pb-2.5">
                  <div className="max-h-44 overflow-auto rounded-md bg-muted/30 p-2">
                    <div className="flex flex-wrap gap-1">
                      {axis.values.map((value) => (
                        <Badge
                          key={`${axis.key}:${value}`}
                          variant="outline"
                          className="max-w-[180px] truncate rounded-md font-mono text-[11px] font-normal text-muted-foreground"
                          title={value}
                        >
                          {value}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </DashboardCard>

      <DashboardCard
        title="Curate"
        description="Reorganize workspace content with an agent"
        className="lg:col-span-6"
      >
        <CurateComposer
          projectId={projectId}
          experimentId={experimentId}
          onComplete={() => onRefresh()}
        />
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
      <EmptyState icon={<WorkflowIcon className="h-6 w-6" />} title="No workflow" />
    </div>
  );

  return (
    <>
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
              onRunCreated={(runId) => {
                onRefresh();
                navigateToRun(runId);
              }}
            />
            <CreateSweepDialog
              projectId={projectId}
              experimentId={experimentId}
              onCreated={() => {
                onRefresh();
                setActiveTab("runs");
              }}
            />
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              aria-label="Agent"
              title="Agent"
              onClick={() =>
                setSelection({
                  objectType: "agent",
                  objectId: "new",
                  scope: { projectId, experimentId },
                })
              }
            >
              <Bot className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleDelete}
              disabled={isDeleting}
              className="h-7 w-7 text-muted-foreground hover:text-destructive"
              aria-label="Delete"
              title="Delete"
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
                      <span className="text-[11px] text-muted-foreground">⇧ range · ⌘ toggle</span>
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
      {confirmDialog}
    </>
  );
};
