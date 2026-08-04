import {
  Archive,
  Copy,
  ExternalLink,
  FlaskConical,
  FolderKanban,
  Play,
  Trash2,
  Workflow,
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";

import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import type { DataTableColumn, DataTableRowAction } from "@/app/components/entity";
import {
  CopyButton,
  DashboardCard,
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
import { successRate } from "@/app/renderers/dashboardData";
import { buildProjectWorkbenchData } from "@/app/renderers/entityWorkbenchData";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, ExperimentSummary, RendererProps } from "@/app/types";
import {
  WorkbenchAction,
  WorkbenchIconAction,
  WorkbenchOperationState,
  WorkbenchRetryAction,
} from "@/components/workbench";
import { formatDateTime } from "@/lib/datetime";

const countAssetsByKind = (assets: ApiAssetResponse[]): Array<[string, number]> => {
  const counts = new Map<string, number>();
  for (const asset of assets) {
    counts.set(asset.kind, (counts.get(asset.kind) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
};

/**
 * Every project tab uses the same grid-backed shell and one continuous work
 * surface. Content may be tabular or dashboard-like, but it never becomes a
 * collection of floating cards.
 */
const ProjectTabSurface = ({ children }: { children: ReactNode }): JSX.Element => (
  <div className="molexp-dashboard flex-1 overflow-auto bg-canvas p-3 sm:p-4">
    <div className="mx-auto min-h-full w-full max-w-7xl bg-surface/95">{children}</div>
  </div>
);

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
        id: "copy-id",
        label: "Copy experiment ID",
        icon: Copy,
        onSelect: () => copyToClipboard(exp.id),
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
          <div className="flex h-control-compact w-control-compact items-center justify-center text-muted-foreground">
            <FlaskConical className="h-3.5 w-3.5" />
          </div>
          <div className="min-w-0">
            <div className="truncate text-body-lg font-medium text-foreground">{exp.name}</div>
            <div className="flex items-center gap-0.5 font-mono text-micro text-muted-foreground">
              <span className="truncate">{exp.id.substring(0, 12)}</span>
              <CopyButton value={exp.id} label="experiment ID" className="size-5" />
            </div>
          </div>
        </div>
      ),
    },
    {
      key: "runs",
      header: "Runs",
      width: "w-56",
      cell: (exp) => {
        const rollup = workbench.experiments.find((item) => item.experiment.id === exp.id);
        if (!rollup || rollup.counts.total === 0) {
          return <span className="text-label text-muted-foreground">No runs</span>;
        }
        return (
          <div className="flex items-center gap-3">
            <span className="w-control-compact font-medium tabular-nums text-foreground">
              {rollup.counts.total}
            </span>
            <div className="min-w-32 flex-1">
              <StatusDistribution counts={rollup.counts} legend={false} />
            </div>
          </div>
        );
      },
    },
    {
      key: "workflow",
      header: "Tasks",
      width: "w-24",
      cell: (exp) => {
        const rollup = workbench.experiments.find((item) => item.experiment.id === exp.id);
        return (
          <span className="inline-flex items-center gap-2 text-label tabular-nums text-muted-foreground">
            <Workflow className="h-3.5 w-3.5" />
            {rollup?.workflowSummary.exists ? rollup.workflowSummary.taskCount : "—"}
          </span>
        );
      },
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-40",
      cell: (exp) => (
        <span className="text-label text-muted-foreground" title={exp.updatedAt}>
          {formatDateTime(exp.updatedAt)}
        </span>
      ),
    },
    {
      key: "action",
      header: "",
      width: "w-14",
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
        <div className="flex items-center gap-2 text-body-lg font-medium text-foreground">
          <Archive className="h-4 w-4 text-muted-foreground" />
          {asset.name}
        </div>
      ),
    },
    {
      key: "kind",
      header: "Kind",
      width: "w-36",
      cell: (asset) => <span className="text-muted-foreground">{asset.kind}</span>,
    },
    {
      key: "scope",
      header: "Scope",
      width: "w-40",
      cell: (asset) => (
        <span className="font-mono text-label text-muted-foreground">
          {asset.scope_kind}
          {asset.scope_ids.length > 0 ? ` · ${asset.scope_ids.join("/")}` : ""}
        </span>
      ),
    },
    {
      key: "size",
      header: "Size",
      width: "w-32",
      cell: (asset) => {
        const size = (asset.extra as Record<string, unknown> | undefined)?.size;
        return (
          <span className="font-mono text-label">
            {typeof size === "number" ? `${size} B` : "—"}
          </span>
        );
      },
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-44",
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
  const projectSuccessRate = successRate(workbench.counts);

  const overviewContent = (
    <ProjectTabSurface>
      <StatGrid className="bg-surface-subtle/55">
        <StatCard label="Experiments" value={projectExperiments.length} />
        <StatCard label="Runs" value={projectRuns.length} muted={projectRuns.length === 0} />
        <StatCard
          label="Success rate"
          value={projectSuccessRate === null ? "—" : `${projectSuccessRate.toFixed(0)}%`}
          tone="success"
          muted={projectSuccessRate === null}
        />
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

      <div className="grid gap-x-6 px-1 py-2 lg:grid-cols-12">
        <DashboardCard
          title="Identity"
          className="bg-transparent lg:col-span-4"
          bodyClassName="space-y-3"
        >
          <MetaGrid columns={2}>
            <MetaField
              label="Project ID"
              value={project.id}
              mono
              title={project.id}
              copyValue={project.id}
            />
            <MetaField label="State" value={project.status} mono />
            <MetaField
              label="Updated"
              value={formatDateTime(project.updatedAt)}
              title={project.updatedAt}
              copyValue={project.updatedAt}
            />
            {project.workspaceKey && (
              <MetaField
                label="Workspace"
                value={project.workspaceKey}
                mono
                copyValue={project.workspaceKey}
              />
            )}
          </MetaGrid>
          {project.summary && (
            <p className="text-body-lg leading-relaxed text-muted-foreground">{project.summary}</p>
          )}
        </DashboardCard>

        <DashboardCard
          title="Run status"
          description={
            workbench.counts.total === 0
              ? "No runs yet"
              : `${workbench.counts.total} run${workbench.counts.total === 1 ? "" : "s"} across experiments`
          }
          className="bg-transparent lg:col-span-4"
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
          className="bg-transparent lg:col-span-4"
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
                <WorkbenchRetryAction
                  onClick={() => {
                    setProjectAssetsLoading(true);
                    setProjectAssetsRequestVersion((version) => version + 1);
                  }}
                />
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
      </div>

      <DashboardCard
        title="Experiment matrix"
        description={`${projectExperiments.length} experiment${projectExperiments.length === 1 ? "" : "s"} · live operational rollup`}
        className="bg-surface-subtle/35"
        bodyClassName="p-0"
      >
        {workbench.experiments.length === 0 ? (
          <p className="px-3 py-4 text-label text-muted-foreground">
            No experiments in this project yet.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <div className="divide-y divide-border/60">
              {workbench.experiments.map((item) => (
                <div
                  key={item.experiment.id}
                  className="group grid min-w-170 grid-cols-(--project-run-grid-columns) items-center gap-3 px-3 py-2 transition-colors hover:bg-interactive/50"
                >
                  <WorkbenchAction
                    kind="ghost"
                    size="content"
                    type="button"
                    className="min-w-0 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    onClick={() => navigateToExperiment(item.experiment.id)}
                  >
                    <span className="block truncate text-body font-medium text-foreground">
                      {item.experiment.name}
                    </span>
                    <span className="block truncate font-mono text-micro text-muted-foreground">
                      {item.experiment.id}
                    </span>
                  </WorkbenchAction>
                  <span className="text-right font-mono text-label tabular-nums text-foreground">
                    {item.counts.total} runs
                  </span>
                  <StatusDistribution counts={item.counts} legend={false} />
                  <span className="text-right font-mono text-label tabular-nums text-muted-foreground">
                    {item.workflowSummary.exists
                      ? `${item.workflowSummary.taskCount} tasks`
                      : "no graph"}
                  </span>
                  <span
                    className="truncate text-right text-micro text-muted-foreground"
                    title={item.experiment.updatedAt}
                  >
                    {formatDateTime(item.experiment.updatedAt)}
                  </span>
                  <CopyButton
                    value={item.experiment.id}
                    label={`${item.experiment.name} ID`}
                    className="size-5"
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </DashboardCard>
    </ProjectTabSurface>
  );

  return (
    <>
      <EntityPage
        icon={FolderKanban}
        title={project.name}
        status={project.status}
        subtitle={project.summary || undefined}
        actions={<CopyButton value={project.id} label="project ID" />}
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
              <ProjectTabSurface>
                <div className="flex min-h-full flex-col" aria-busy={deletingExperimentId !== null}>
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
                        <WorkbenchRetryAction
                          onClick={() =>
                            void handleDeleteExperiment(experimentDeleteError.experiment)
                          }
                        />
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
              </ProjectTabSurface>
            ),
          },
          {
            value: "assets",
            label: "Assets",
            content: (
              <ProjectTabSurface>
                <div className="flex min-h-full flex-col">
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
                        <WorkbenchRetryAction
                          onClick={() => {
                            setProjectAssetsLoading(true);
                            setProjectAssetsRequestVersion((version) => version + 1);
                          }}
                        />
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
                </div>
              </ProjectTabSurface>
            ),
          },
          {
            value: "settings",
            label: "Settings",
            content: (
              <ProjectTabSurface>
                <div className="flex min-h-full">
                  <nav
                    aria-label="Project settings categories"
                    className="hidden w-44 shrink-0 bg-surface-subtle/45 p-2 sm:flex sm:flex-col sm:gap-0.5"
                  >
                    <WorkbenchAction
                      kind="ghost"
                      size="content"
                      type="button"
                      className="flex w-full items-center gap-2 border-l-2 border-accent bg-accent-muted/50 px-2.5 py-1.5 text-left text-micro font-medium text-foreground"
                      onClick={() =>
                        document.getElementById("project-settings-general")?.scrollIntoView()
                      }
                    >
                      <FolderKanban className="size-3.5 text-accent" aria-hidden />
                      Project
                    </WorkbenchAction>
                    <WorkbenchAction
                      kind="ghost"
                      size="content"
                      type="button"
                      className="flex w-full items-center gap-2 border-l-2 border-transparent px-2.5 py-1.5 text-left text-micro text-muted-foreground transition-colors hover:bg-interactive hover:text-foreground"
                      onClick={() =>
                        document.getElementById("project-settings-lifecycle")?.scrollIntoView()
                      }
                    >
                      <Trash2 className="size-3.5" aria-hidden />
                      Lifecycle
                    </WorkbenchAction>
                  </nav>

                  <div className="min-w-0 flex-1 px-5 py-4 sm:px-6">
                    <section id="project-settings-general" className="space-y-3 py-3">
                      <h3 className="text-body font-semibold tracking-tight text-foreground">
                        Project
                      </h3>
                      <dl className="space-y-1">
                        <div className="flex min-h-control-compact items-center justify-between gap-4 px-0.5">
                          <dt className="text-micro text-muted-foreground">Name</dt>
                          <dd className="truncate text-body text-foreground">{project.name}</dd>
                        </div>
                        <div className="flex min-h-control-compact items-center justify-between gap-4 px-0.5">
                          <dt className="text-micro text-muted-foreground">Identifier</dt>
                          <dd className="flex min-w-0 items-center gap-1 font-mono text-label text-foreground">
                            <span className="truncate">{project.id}</span>
                            <CopyButton value={project.id} label="project ID" />
                          </dd>
                        </div>
                        <div className="flex min-h-control-compact items-center justify-between gap-4 px-0.5">
                          <dt className="text-micro text-muted-foreground">Contents</dt>
                          <dd className="text-label text-foreground">
                            {projectExperiments.length} experiments · {projectRuns.length} runs
                          </dd>
                        </div>
                      </dl>
                    </section>

                    <section id="project-settings-lifecycle" className="space-y-3 py-7">
                      <h3 className="text-body font-semibold tracking-tight text-foreground">
                        Lifecycle
                      </h3>
                      <div className="flex items-center justify-between gap-4 px-0.5 py-1.5 hover:bg-interactive/40">
                        <div className="min-w-0">
                          <p className="text-body text-foreground">Delete project</p>
                          <p className="mt-0.5 text-micro leading-relaxed text-muted-foreground">
                            Remove this project and its experiment and run hierarchy from the
                            workspace view. This cannot be undone from the UI.
                          </p>
                        </div>
                        <WorkbenchAction
                          kind="danger"
                          size="compact"
                          onClick={handleDelete}
                          disabled={isDeleting}
                          aria-busy={isDeleting}
                        >
                          {isDeleting ? "Deleting…" : "Delete project"}
                        </WorkbenchAction>
                      </div>

                      {isDeleting && (
                        <WorkbenchOperationState
                          kind="running"
                          density="compact"
                          title="Deleting project…"
                          detail={projectId}
                        />
                      )}
                      {deleteError && (
                        <WorkbenchOperationState
                          kind="error"
                          density="compact"
                          title="Could not delete project"
                          detail={deleteError}
                          action={<WorkbenchRetryAction onClick={() => void handleDelete()} />}
                        />
                      )}
                    </section>
                  </div>
                </div>
              </ProjectTabSurface>
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
