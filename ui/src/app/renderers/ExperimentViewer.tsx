import {
  Ban,
  BarChart3,
  Bot,
  Check,
  FileQuestion,
  FlaskConical,
  ListChecks,
  Play,
  SlidersHorizontal,
  Trash2,
  Workflow as WorkflowIcon,
  X,
} from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import { CreateSweepDialog } from "@/app/components/CreateSweepDialog";
import { CurateComposer } from "@/app/components/CurateComposer";
import type { DataTableColumn, DataTableRowAction } from "@/app/components/entity";
import {
  CopyButton,
  DashboardCard,
  DashboardGrid,
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityPage,
  MetaField,
  MetaGrid,
  MiniBars,
  ParamChip,
  StatCard,
  StatGrid,
  StatusDistribution,
  StatusIcon,
} from "@/app/components/entity";
// Module-path import (not the barrel) — see RunViewer.tsx for the loader rationale.
import { KnowledgeBacklinksCard } from "@/app/components/entity/KnowledgeBacklinksCard";
import {
  countRunStatuses,
  formatDuration,
  formatScalar,
  successRate,
} from "@/app/renderers/dashboardData";
import { ExperimentCompare } from "@/app/renderers/ExperimentCompare";
import { buildExperimentWorkbenchData } from "@/app/renderers/entityWorkbenchData";
import { WorkflowGraphViewer } from "@/app/renderers/WorkflowGraphViewer";
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
import { Code as InlineCode } from "@/components/ui/code";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { toast } from "@/components/ui/toast";
import {
  WorkbenchAction,
  WorkbenchIconAction,
  WorkbenchTag,
  WorkbenchToggleAction,
} from "@/components/workbench";
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
  if (entries.length === 0) return <span className="text-label text-muted-foreground">—</span>;
  const visible = entries.slice(0, 3);
  return (
    <div className="flex max-w-80 flex-wrap items-center gap-1">
      {visible.map(([key, value]) => (
        <ParamChip key={key} name={key} value={formatScalar(value)} />
      ))}
      {entries.length > visible.length && (
        <Popover>
          <PopoverTrigger asChild>
            <WorkbenchAction kind="ghost" size="compact" className="h-6 px-2 text-micro">
              +{entries.length - visible.length}
            </WorkbenchAction>
          </PopoverTrigger>
          <PopoverContent side="bottom" align="start" className="max-h-80 w-72 overflow-auto p-3">
            <dl className="space-y-2">
              {entries.map(([key, value]) => (
                <div
                  key={key}
                  className="grid grid-cols-(--experiment-meta-grid-columns) gap-2 text-label"
                >
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
      <CopyButton
        value={JSON.stringify(run.parameters ?? {}, null, 2)}
        label={`${run.name || run.id} parameters`}
        className="size-5"
      />
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
            Stop{" "}
            <InlineCode className="rounded-control bg-muted px-1 py-1 text-label">
              {run.id}
            </InlineCode>
            ?
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
      width: "w-44",
      cell: (run) => (
        <div className="min-w-0">
          <div className="truncate text-body-lg font-medium text-foreground">
            {run.name || run.id}
          </div>
          <div className="flex items-center gap-0.5 font-mono text-micro text-muted-foreground">
            <span className="truncate">{run.id.substring(0, 12)}</span>
            <CopyButton value={run.id} label="run ID" className="size-5" />
          </div>
        </div>
      ),
    },
    {
      key: "status",
      header: "State",
      width: "w-18",
      cell: (run) => <StatusIcon status={run.status} />,
    },
    {
      key: "parameters",
      header: "Parameters",
      width: "w-90",
      cell: (run) => <ParametersCell run={run} keys={parameterKeys} />,
    },
    {
      key: "result",
      header: "Result",
      cell: (run) => {
        const preview = formatResultPreview(run.results ?? {});
        const full = JSON.stringify(run.results ?? {}, null, 2);
        return (
          <div className="flex max-w-72 items-center gap-1">
            <span
              className="min-w-0 flex-1 truncate font-mono text-label text-foreground"
              title={full}
            >
              {preview}
            </span>
            <CopyButton value={full} label={`${run.name || run.id} results`} className="size-5" />
          </div>
        );
      },
    },
    {
      key: "duration",
      header: "Duration",
      width: "w-24",
      cell: (run) => {
        const d = formatDuration(run.startedAt, run.finishedAt);
        return (
          <span className="font-mono text-label text-muted-foreground">
            {d ?? <span className="text-muted-foreground">—</span>}
          </span>
        );
      },
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-40",
      cell: (run) => (
        <span className="text-label text-muted-foreground" title={run.updatedAt}>
          {formatDateTime(run.updatedAt)}
        </span>
      ),
    },
    {
      key: "action",
      header: "",
      width: "w-24",
      align: "right",
      cell: (run) => {
        const verb = primaryRunVerb(run.status);
        if (!verb) return null;
        return (
          <WorkbenchIconAction
            label={verb.label}
            className={`size-6 ${verb.kind === "cancel" ? "text-destructive hover:bg-destructive/10 hover:text-destructive" : ""}`}
            onClick={(event) => {
              event.stopPropagation();
              if (verb.kind === "start") navigateToRunView(run);
              else if (verb.kind === "cancel") void handleCancelRun(run);
              else if (verb.kind === "resume") void handleResumeRun(run);
            }}
          >
            {verb.kind === "cancel" ? <Ban className="size-3.5" /> : <Play className="size-3.5" />}
          </WorkbenchIconAction>
        );
      },
    },
  ];

  // Leading tick column, shown only in multi-select mode. The cell button reads
  // the native event so shift (range) / ctrl|meta (toggle) modifiers reach the
  // pure selection reducer — DataTable's row activation carries no native event.
  const selectionColumn: DataTableColumn<RunSummary> = {
    key: "select",
    header: "",
    width: "w-control-comfortable",
    cell: (run) => {
      const checked = multi.selected.has(run.id);
      return (
        <WorkbenchAction
          kind="ghost"
          size="content"
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
          className={`flex h-4 w-4 items-center justify-center rounded-control border transition-colors ${
            checked
              ? "border-accent bg-accent text-accent-foreground"
              : "border-border hover:border-accent"
          }`}
        >
          {checked && <Check className="h-3 w-3" />}
        </WorkbenchAction>
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
  const experimentSuccessRate = successRate(counts);

  const overviewContent = (
    <DashboardGrid>
      <div className="lg:col-span-12">
        <StatGrid>
          <StatCard label="Runs" value={counts.total} muted={counts.total === 0} />
          <StatCard
            label="Success rate"
            value={experimentSuccessRate === null ? "—" : `${experimentSuccessRate.toFixed(0)}%`}
            tone="success"
            muted={experimentSuccessRate === null}
          />
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

      <DashboardCard
        title="Identity"
        className="lg:col-span-4"
        action={
          <CopyButton
            value={JSON.stringify(
              { projectId: experiment.projectId, experimentId: experiment.id },
              null,
              2,
            )}
            label="experiment coordinates"
          />
        }
      >
        <MetaGrid columns={2}>
          <MetaField
            label="Experiment ID"
            value={experiment.id}
            mono
            title={experiment.id}
            copyValue={experiment.id}
          />
          <MetaField label="Project" value={project?.name ?? projectId} copyValue={projectId} />
          <MetaField
            label="Updated"
            value={formatDateTime(experiment.updatedAt)}
            title={experiment.updatedAt}
            copyValue={experiment.updatedAt}
          />
          <MetaField
            label="Workflow tasks"
            value={workbench.workflowSummary.exists ? workbench.workflowSummary.taskCount : "—"}
          />
          <MetaField
            label="Workflow file"
            value={experiment.workflowFile || "—"}
            mono
            title={experiment.workflowFile || undefined}
            copyValue={experiment.workflowFile || undefined}
          />
          {experiment.planRunId && (
            <MetaField
              label="Plan run"
              value={experiment.planRunId}
              mono
              title={experiment.planRunId}
              copyValue={experiment.planRunId}
            />
          )}
        </MetaGrid>
      </DashboardCard>

      <DashboardCard
        title="Run status"
        description={
          counts.total === 0
            ? "No runs yet"
            : `${counts.total} run${counts.total === 1 ? "" : "s"} in this experiment`
        }
        className="lg:col-span-4"
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
            <span className="inline-flex items-center gap-2 text-label text-muted-foreground">
              <WorkflowIcon className="h-3.5 w-3.5" />
              Graph
            </span>
          ) : undefined
        }
      >
        {workflowGraph ? (
          <WorkflowGraph ir={workflowGraph} height={240} />
        ) : (
          <p className="text-body-lg text-muted-foreground">No workflow graph recorded.</p>
        )}
      </DashboardCard>

      <DashboardCard
        title="Run groups"
        description={
          workbench.parameterAxes.some((axis) => axis.count > 1)
            ? "Grouped by the first varying parameter"
            : "Grouped by execution state"
        }
        className="lg:col-span-4"
      >
        <MiniBars
          data={workbench.runGroups.slice(0, 8).map((group) => ({
            label: group.label,
            value: group.runs.length,
            hint: `${group.counts.succeeded}/${group.counts.total}`,
            onClick: () => setActiveTab("runs"),
          }))}
          emptyLabel="No runs available to group."
        />
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
          <p className="text-body-lg text-muted-foreground">No parameter axes declared.</p>
        ) : (
          <Accordion type="multiple" className="border-y border-border">
            {workbench.parameterAxes.map((axis) => (
              <AccordionItem key={axis.key} value={axis.key} className="border-border px-3">
                <AccordionTrigger className="py-3 text-label hover:no-underline">
                  <span className="inline-flex min-w-0 items-center gap-2">
                    <SlidersHorizontal className="h-3.5 w-3.5 flex-none text-muted-foreground" />
                    <span className="truncate font-medium text-foreground">{axis.key}</span>
                  </span>
                  <WorkbenchTag className="ml-auto mr-2 font-mono text-micro">
                    {axis.count}
                  </WorkbenchTag>
                </AccordionTrigger>
                <AccordionContent className="pb-3">
                  <div className="max-h-44 overflow-auto rounded-control bg-muted/30 p-2">
                    <div className="flex flex-wrap gap-1">
                      {axis.values.map((value) => (
                        <WorkbenchTag
                          key={`${axis.key}:${value}`}
                          meaning="metadata"
                          className="max-w-44 truncate rounded-control font-mono text-micro font-normal text-muted-foreground"
                          title={value}
                        >
                          {value}
                        </WorkbenchTag>
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
            <CopyButton value={experiment.id} label="experiment ID" />
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
            <WorkbenchIconAction
              label="Agent"
              kind="ghost"
              className="h-control-compact w-control-compact"
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
            </WorkbenchIconAction>
            <WorkbenchIconAction
              label="Delete"
              kind="ghost"
              onClick={handleDelete}
              disabled={isDeleting}
              className="h-control-compact w-control-compact text-muted-foreground hover:text-destructive"
              aria-label="Delete"
              title="Delete"
            >
              <Trash2 className="h-4 w-4" />
            </WorkbenchIconAction>
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
                <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
                  <WorkbenchToggleAction
                    label={multi.enabled ? "Exit multi-select" : "Select multiple runs"}
                    pressed={multi.enabled}
                    onClick={multi.toggleMode}
                  >
                    <ListChecks className="h-3.5 w-3.5" />
                  </WorkbenchToggleAction>
                  {multi.enabled && (
                    <>
                      <span className="text-label text-muted-foreground">
                        {multi.selected.size} selected
                      </span>
                      <WorkbenchIconAction
                        label="Compare selected runs"
                        disabled={multi.selected.size === 0}
                        onClick={() => setActiveTab("compare")}
                      >
                        <BarChart3 className="h-3.5 w-3.5" />
                      </WorkbenchIconAction>
                      <WorkbenchIconAction
                        label="Clear run selection"
                        disabled={multi.selected.size === 0}
                        onClick={multi.clear}
                      >
                        <X className="h-3.5 w-3.5" />
                      </WorkbenchIconAction>
                      <span className="text-micro text-muted-foreground">⇧ range · ⌘ toggle</span>
                    </>
                  )}
                </div>
                <DataTable
                  columns={tableColumns}
                  data={runs}
                  getRowKey={(run) => run.id}
                  getRowLabel={(run) =>
                    multi.enabled
                      ? `${multi.selected.has(run.id) ? "Deselect" : "Select"} run ${run.name || run.id}`
                      : `Open run ${run.name || run.id}`
                  }
                  onRowActivate={
                    multi.enabled
                      ? (run) =>
                          multi.selectAt(runIndex.get(run.id) ?? 0, { shift: false, meta: true })
                      : (run) => navigateToRun(run.id)
                  }
                  rowActions={runRowActions}
                  rowClassName={(run) =>
                    multi.enabled && multi.selected.has(run.id) ? "bg-accent/5" : ""
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
        ]}
      />
      {confirmDialog}
    </>
  );
};
