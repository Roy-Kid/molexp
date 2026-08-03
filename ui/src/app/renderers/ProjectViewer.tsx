import {
  Archive,
  Copy,
  ExternalLink,
  FlaskConical,
  FolderKanban,
  Play,
  Workflow,
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
  EntityPage,
  MetaField,
  MetaGrid,
  MiniBars,
  StatCard,
  StatGrid,
  StatusDistribution,
} from "@/app/components/entity";
import { buildProjectWorkbenchData } from "@/app/renderers/entityWorkbenchData";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, ExperimentSummary, RendererProps } from "@/app/types";
import {
  WorkbenchAction,
  WorkbenchIconAction,
  WorkbenchOperationState,
} from "@/components/workbench";
import { formatDateTime } from "@/lib/datetime";

const countAssetsByKind = (assets: ApiAssetResponse[]): Array<[string, number]> => {
  const counts = new Map<string, number>();
  for (const asset of assets) {
    counts.set(asset.kind, (counts.get(asset.kind) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
};

export const ProjectViewer = ({ selection, snapshot, onRefresh }: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deletingExperimentId, setDeletingExperimentId] = useState<string | null>(null);
  const [experimentDeleteError, setExperimentDeleteError] = useState<{
    experiment: ExperimentSummary;
    message: string;
  } | null>(null);
  const [projectAssets, setProjectAssets] = useState<ApiAssetResponse[]>([]);
  const [projectAssetsLoading, setProjectAssetsLoading] = useState(true);
  const [projectAssetsError, setProjectAssetsError] = useState<string | null>(null);
  const [settledProjectAssetsId, setSettledProjectAssetsId] = useState<string | null>(null);
  const [projectAssetsRequestVersion, setProjectAssetsRequestVersion] = useState(0);
  const [createRunExperimentId, setCreateRunExperimentId] = useState<string | null>(null);
  const { setSelection } = useNavigationState(snapshot);

  const projectId = selection.objectId;
  const project = snapshot.projects.find((p) => p.id === projectId);

  useEffect(() => {
    void projectAssetsRequestVersion;
    if (!projectId) {
      setProjectAssets([]);
      setProjectAssetsLoading(false);
      setProjectAssetsError(null);
      setSettledProjectAssetsId(null);
      return;
    }

    let cancelled = false;
    setProjectAssets([]);
    setProjectAssetsLoading(true);
    setProjectAssetsError(null);
    workspaceApi
      .getProjectAssets(projectId)
      .then((assets) => {
        if (!cancelled) setProjectAssets(assets);
      })
      .catch((err) => {
        if (!cancelled) {
          setProjectAssetsError(
            err instanceof Error ? err.message : "Failed to load project assets",
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setProjectAssetsLoading(false);
          setSettledProjectAssetsId(projectId);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, projectAssetsRequestVersion]);

  const projectExperiments = useMemo(
    () => snapshot.experiments.filter((e) => e.projectId === projectId),
    [snapshot.experiments, projectId],
  );

  const projectRuns = useMemo(
    () => snapshot.runs.filter((r) => r.projectId === projectId),
    [snapshot.runs, projectId],
  );

  const workbench = useMemo(
    () => buildProjectWorkbenchData(projectId, snapshot, projectAssets),
    [projectId, snapshot, projectAssets],
  );
  const projectAssetsByKind = useMemo(() => countAssetsByKind(projectAssets), [projectAssets]);
  const projectAssetsPending = projectAssetsLoading || settledProjectAssetsId !== projectId;

  const handleDelete = async () => {
    if (!confirm(`Are you sure you want to delete project "${projectId}"?`)) {
      return;
    }
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await workspaceApi.deleteProject(projectId);
      onRefresh();
      setSelection(null);
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : "Failed to delete project");
    } finally {
      setIsDeleting(false);
    }
  };

  const navigateToExperiment = (experimentId: string) => {
    setSelection({
      objectType: "experiment",
      objectId: experimentId,
    });
  };

  const handleDeleteExperiment = async (experiment: ExperimentSummary) => {
    if (!confirm(`Are you sure you want to delete experiment "${experiment.id}"?`)) {
      return;
    }
    setDeletingExperimentId(experiment.id);
    setExperimentDeleteError(null);
    try {
      await workspaceApi.deleteExperiment(experiment.projectId, experiment.id);
      onRefresh();
    } catch (error) {
      setExperimentDeleteError({
        experiment,
        message: error instanceof Error ? error.message : "Failed to delete experiment",
      });
    } finally {
      setDeletingExperimentId(null);
    }
  };

  const copyToClipboard = (text: string) => {
    void navigator.clipboard.writeText(text);
  };

  const experimentRowActions = (
    exp: ExperimentSummary,
  ): DataTableRowAction<ExperimentSummary>[] => {
    const workflow = snapshot.workflows.find((item) => item.experimentId === exp.id);
    return [
      {
        id: "open",
        label: "Open experiment",
        icon: ExternalLink,
        onSelect: () => navigateToExperiment(exp.id),
      },
      {
        id: "new-run",
        label: "New run",
        icon: Play,
        onSelect: () => setCreateRunExperimentId(exp.id),
      },
      {
        id: "open-workflow",
        label: "Open workflow",
        icon: Workflow,
        disabled: !workflow,
        title: workflow ? undefined : "No workflow is configured for this experiment.",
        onSelect: () => {
          if (workflow) {
            setSelection({
              objectType: "workflow",
              objectId: workflow.id,
              workflowId: workflow.id,
            });
          }
        },
      },
      {
        id: "delete",
        label: "Delete experiment",
        disabled: deletingExperimentId === exp.id,
        title: deletingExperimentId === exp.id ? "This experiment is being deleted." : undefined,
        destructive: true,
        separatorBefore: true,
        onSelect: (experiment) => {
          void handleDeleteExperiment(experiment);
        },
      },
    ];
  };

  const assetRowActions = (asset: ApiAssetResponse): DataTableRowAction<ApiAssetResponse>[] => [
    {
      id: "open",
      label: "Open asset",
      icon: ExternalLink,
      onSelect: () => setSelection({ objectType: "asset", objectId: asset.id }),
    },
    {
      id: "copy-id",
      label: "Copy asset ID",
      icon: Copy,
      onSelect: () => copyToClipboard(asset.id),
    },
  ];

  if (!project) {
    return (
      <WorkbenchOperationState
        kind="empty"
        title="Project not found"
        detail="The current workspace snapshot no longer contains this project."
      />
    );
  }

  const experimentColumns: DataTableColumn<ExperimentSummary>[] = [
    {
      key: "name",
      header: "Experiment",
      cell: (exp) => (
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center text-muted-foreground">
            <FlaskConical className="h-3.5 w-3.5" />
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-foreground">{exp.name}</div>
            <div className="truncate font-mono text-micro text-muted-foreground">
              {exp.id.substring(0, 8)}
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "runs",
      header: "Runs",
      width: "w-[220px]",
      cell: (exp) => {
        const rollup = workbench.experiments.find((item) => item.experiment.id === exp.id);
        if (!rollup || rollup.counts.total === 0) {
          return <span className="text-xs text-muted-foreground">No runs</span>;
        }
        return (
          <div className="flex items-center gap-3">
            <span className="w-7 font-medium tabular-nums text-foreground">
              {rollup.counts.total}
            </span>
            <div className="min-w-[120px] flex-1">
              <StatusDistribution counts={rollup.counts} legend={false} />
            </div>
          </div>
        );
      },
    },
    {
      key: "workflow",
      header: "Tasks",
      width: "w-[90px]",
      cell: (exp) => {
        const rollup = workbench.experiments.find((item) => item.experiment.id === exp.id);
        return (
          <span className="inline-flex items-center gap-2 text-xs tabular-nums text-muted-foreground">
            <Workflow className="h-3.5 w-3.5" />
            {rollup?.workflowSummary.exists ? rollup.workflowSummary.taskCount : "—"}
          </span>
        );
      },
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-[160px]",
      cell: (exp) => (
        <span className="text-xs text-muted-foreground" title={exp.updatedAt}>
          {formatDateTime(exp.updatedAt)}
        </span>
      ),
    },
    {
      key: "action",
      header: "",
      width: "w-[52px]",
      align: "right",
      cell: (exp) => (
        <WorkbenchIconAction
          label={`New run in ${exp.name}`}
          kind="ghost"
          size="default"
          className="opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100"
          onClick={(event) => {
            event.stopPropagation();
            setCreateRunExperimentId(exp.id);
          }}
        >
          <Play className="h-4 w-4 text-muted-foreground hover:text-foreground" />
        </WorkbenchIconAction>
      ),
    },
  ];

  const assetColumns: DataTableColumn<ApiAssetResponse>[] = [
    {
      key: "name",
      header: "Name",
      cell: (asset) => (
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Archive className="h-4 w-4 text-muted-foreground" />
          {asset.name}
        </div>
      ),
    },
    {
      key: "kind",
      header: "Kind",
      width: "w-[140px]",
      cell: (asset) => <span className="text-muted-foreground">{asset.kind}</span>,
    },
    {
      key: "scope",
      header: "Scope",
      width: "w-[160px]",
      cell: (asset) => (
        <span className="font-mono text-xs text-muted-foreground">
          {asset.scope_kind}
          {asset.scope_ids.length > 0 ? ` · ${asset.scope_ids.join("/")}` : ""}
        </span>
      ),
    },
    {
      key: "size",
      header: "Size",
      width: "w-[120px]",
      cell: (asset) => {
        const size = (asset.extra as Record<string, unknown> | undefined)?.size;
        return (
          <span className="font-mono text-xs">{typeof size === "number" ? `${size} B` : "—"}</span>
        );
      },
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-[180px]",
      cell: (asset) => (
        <span className="text-muted-foreground" title={asset.updated_at}>
          {formatDateTime(asset.updated_at)}
        </span>
      ),
    },
  ];

  const createRunExperiment = createRunExperimentId
    ? snapshot.experiments.find((experiment) => experiment.id === createRunExperimentId)
    : null;

  const overviewContent = (
    <DashboardGrid>
      <div className="lg:col-span-12">
        <StatGrid>
          <StatCard label="Experiments" value={projectExperiments.length} />
          <StatCard label="Runs" value={projectRuns.length} muted={projectRuns.length === 0} />
          <StatCard
            label="Running"
            value={workbench.counts.running}
            tone="running"
            muted={workbench.counts.running === 0}
          />
          <StatCard
            label="Failed"
            value={workbench.counts.failed}
            tone="error"
            muted={workbench.counts.failed === 0}
          />
          <StatCard
            label="Assets"
            value={projectAssetsPending ? "—" : projectAssetsError ? "!" : projectAssets.length}
            hint={projectAssetsPending ? "Loading" : projectAssetsError ? "Unavailable" : undefined}
            tone={projectAssetsError ? "error" : "neutral"}
            muted={!projectAssetsPending && !projectAssetsError && projectAssets.length === 0}
          />
        </StatGrid>
      </div>

      <DashboardCard title="Identity" className="lg:col-span-5" bodyClassName="space-y-4">
        <MetaGrid columns={2}>
          <MetaField label="Project ID" value={project.id} mono title={project.id} />
          <MetaField
            label="Updated"
            value={formatDateTime(project.updatedAt)}
            title={project.updatedAt}
          />
        </MetaGrid>
        {project.summary && (
          <p className="text-sm leading-relaxed text-muted-foreground">{project.summary}</p>
        )}
      </DashboardCard>

      <DashboardCard
        title="Run status"
        description={
          workbench.counts.total === 0
            ? "No runs yet"
            : `${workbench.counts.total} run${workbench.counts.total === 1 ? "" : "s"} across experiments`
        }
        className="lg:col-span-7"
      >
        <StatusDistribution counts={workbench.counts} />
      </DashboardCard>

      <DashboardCard
        title="Assets by kind"
        description={
          projectAssetsPending
            ? "Loading asset registry"
            : projectAssetsError
              ? "Asset registry unavailable"
              : projectAssets.length === 0
                ? "Nothing registered"
                : `${projectAssets.length} registered asset${projectAssets.length === 1 ? "" : "s"}`
        }
        className="lg:col-span-12"
      >
        {projectAssetsPending ? (
          <WorkbenchOperationState
            kind="loading"
            density="compact"
            title="Loading project assets…"
            skeletonRows={3}
          />
        ) : projectAssetsError ? (
          <WorkbenchOperationState
            kind="error"
            density="compact"
            title="Could not load project assets"
            detail={projectAssetsError}
            action={
              <WorkbenchAction
                kind="secondary"
                size="compact"
                onClick={() => {
                  setProjectAssetsLoading(true);
                  setProjectAssetsRequestVersion((version) => version + 1);
                }}
              >
                Retry
              </WorkbenchAction>
            }
          />
        ) : (
          <MiniBars
            data={projectAssetsByKind.slice(0, 8).map(([kind, count]) => ({
              label: kind,
              value: count,
            }))}
            emptyLabel="No assets registered under this project."
          />
        )}
      </DashboardCard>
    </DashboardGrid>
  );

  return (
    <>
      <EntityPage
        icon={FolderKanban}
        title={project.name}
        subtitle={project.summary || undefined}
        tabs={[
          {
            value: "overview",
            label: "Overview",
            content: overviewContent,
          },
          {
            value: "experiments",
            label: "Experiments",
            content: (
              <div aria-busy={deletingExperimentId !== null}>
                {deletingExperimentId && (
                  <WorkbenchOperationState
                    kind="running"
                    density="compact"
                    title="Deleting experiment…"
                    detail={deletingExperimentId}
                  />
                )}
                {experimentDeleteError && (
                  <WorkbenchOperationState
                    kind="error"
                    density="compact"
                    title="Could not delete experiment"
                    detail={experimentDeleteError.message}
                    action={
                      <WorkbenchAction
                        kind="secondary"
                        size="compact"
                        onClick={() =>
                          void handleDeleteExperiment(experimentDeleteError.experiment)
                        }
                      >
                        Retry
                      </WorkbenchAction>
                    }
                  />
                )}
                <DataTable
                  columns={experimentColumns}
                  data={projectExperiments}
                  getRowKey={(exp) => exp.id}
                  getRowLabel={(exp) => `Open experiment ${exp.name}`}
                  onRowActivate={(exp) => navigateToExperiment(exp.id)}
                  rowActions={experimentRowActions}
                  empty={
                    <EmptyState
                      title={EMPTY_COPY.experiments.title}
                      description={EMPTY_COPY.experiments.description}
                    />
                  }
                />
              </div>
            ),
          },
          {
            value: "assets",
            label: "Assets",
            content: (
              <>
                {projectAssetsPending ? (
                  <WorkbenchOperationState
                    kind="loading"
                    title="Loading project assets…"
                    skeletonRows={5}
                  />
                ) : projectAssetsError ? (
                  <WorkbenchOperationState
                    kind="error"
                    title="Could not load project assets"
                    detail={projectAssetsError}
                    action={
                      <WorkbenchAction
                        kind="secondary"
                        size="compact"
                        onClick={() => {
                          setProjectAssetsLoading(true);
                          setProjectAssetsRequestVersion((version) => version + 1);
                        }}
                      >
                        Retry
                      </WorkbenchAction>
                    }
                  />
                ) : (
                  <DataTable
                    columns={assetColumns}
                    data={projectAssets}
                    getRowKey={(asset) => asset.id}
                    getRowLabel={(asset) => `Open asset ${asset.name}`}
                    onRowActivate={(asset) =>
                      setSelection({ objectType: "asset", objectId: asset.id })
                    }
                    rowActions={assetRowActions}
                    empty={<EmptyState title={EMPTY_COPY.assets.title} />}
                  />
                )}
              </>
            ),
          },
          {
            value: "settings",
            label: "Settings",
            content: (
              <div className="overflow-auto p-4 md:p-6">
                <DashboardCard
                  title="Delete project"
                  description="Removes this project and its experiment / run hierarchy from the workspace view."
                  variant="destructive"
                  className="max-w-xl"
                >
                  {isDeleting && (
                    <WorkbenchOperationState
                      kind="running"
                      density="compact"
                      title="Deleting project…"
                      detail={projectId}
                      className="mb-3"
                    />
                  )}
                  {deleteError && (
                    <WorkbenchOperationState
                      kind="error"
                      density="compact"
                      title="Could not delete project"
                      detail={deleteError}
                      action={
                        <WorkbenchAction
                          kind="secondary"
                          size="compact"
                          onClick={() => void handleDelete()}
                        >
                          Retry
                        </WorkbenchAction>
                      }
                    />
                  )}
                  <WorkbenchAction
                    kind="danger"
                    size="default"
                    onClick={handleDelete}
                    disabled={isDeleting}
                    aria-busy={isDeleting}
                  >
                    {isDeleting ? "Deleting…" : "Delete project"}
                  </WorkbenchAction>
                </DashboardCard>
              </div>
            ),
          },
        ]}
      />
      {createRunExperiment && (
        <CreateRunDialog
          projectId={createRunExperiment.projectId}
          experimentId={createRunExperiment.id}
          workflowFile={createRunExperiment.workflowFile || ""}
          open
          trigger={null}
          onOpenChange={(nextOpen) => {
            if (!nextOpen) setCreateRunExperimentId(null);
          }}
          onRunCreated={(runId) => {
            onRefresh();
            setCreateRunExperimentId(null);
            setSelection({ objectType: "run", objectId: runId });
          }}
        />
      )}
    </>
  );
};
