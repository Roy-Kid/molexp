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
} from "@/app/components/entity";
import {
  buildProjectWorkbenchData,
  type ExperimentRollup,
} from "@/app/renderers/entityWorkbenchData";
import { STATUS_GROUPS } from "@/app/runs/statusGroups";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type {
  ApiAssetResponse,
  ExperimentSummary,
  RendererProps,
  SemanticStatus,
} from "@/app/types";
import { Button } from "@/components/ui/button";

const StatusDistributionBar = ({ counts }: { counts: ExperimentRollup["counts"] }): JSX.Element => {
  if (counts.total === 0) {
    return <div className="h-1.5 rounded-full bg-muted" />;
  }
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

const statusTextClass = (status: SemanticStatus): string => {
  switch (status) {
    case "active":
    case "approved":
    case "succeeded":
      return "font-medium text-success";
    case "failed":
    case "rejected":
      return "font-medium text-destructive";
    case "running":
      return "font-medium text-info";
    case "draft":
    case "expired":
    case "waiting_for_review":
      return "font-medium text-warning";
    case "archived":
    case "cancelled":
    case "skipped":
    case "pending":
      return "text-muted-foreground";
  }
};

const countAssetsByKind = (assets: ApiAssetResponse[]): Array<[string, number]> => {
  const counts = new Map<string, number>();
  for (const asset of assets) {
    counts.set(asset.kind, (counts.get(asset.kind) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
};

export const ProjectViewer = ({ selection, snapshot, onRefresh }: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [projectAssets, setProjectAssets] = useState<ApiAssetResponse[]>([]);
  const [createRunExperimentId, setCreateRunExperimentId] = useState<string | null>(null);
  const { setSelection } = useNavigationState(snapshot);

  const projectId = selection.objectId;
  const project = snapshot.projects.find((p) => p.id === projectId);

  useEffect(() => {
    if (projectId) {
      workspaceApi
        .getProjectAssets(projectId)
        .then(setProjectAssets)
        .catch((err) => console.error("Failed to load project assets", err));
    }
  }, [projectId]);

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

  const handleDelete = async () => {
    if (!confirm(`Are you sure you want to delete project "${projectId}"?`)) {
      return;
    }
    setIsDeleting(true);
    try {
      await workspaceApi.deleteProject(projectId);
      onRefresh();
    } catch (error) {
      console.error("Failed to delete project:", error);
      alert("Failed to delete project");
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
    try {
      await workspaceApi.deleteExperiment(experiment.projectId, experiment.id);
      onRefresh();
    } catch (error) {
      console.error("Failed to delete experiment:", error);
      alert("Failed to delete experiment");
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

  if (!project) return <div className="p-8 text-muted-foreground">Project not found.</div>;

  const experimentColumns: DataTableColumn<ExperimentSummary>[] = [
    {
      key: "name",
      header: "Experiment Name",
      cell: (exp) => (
        <div className="flex items-center gap-3">
          <div className="rounded-md bg-purple-500/10 p-1.5 text-purple-600 transition-colors group-hover:bg-purple-500/20">
            <FlaskConical className="h-4 w-4" />
          </div>
          <span className={`font-medium ${statusTextClass(exp.status)}`}>{exp.name}</span>
        </div>
      ),
    },
    {
      key: "id",
      header: "ID",
      width: "w-[120px]",
      cell: (exp) => (
        <span className="font-mono text-xs text-muted-foreground">{exp.id.substring(0, 8)}</span>
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
          <div className="flex items-center gap-2">
            <span className="w-8 font-semibold tabular-nums text-foreground">
              {rollup.counts.total}
            </span>
            <div className="min-w-[120px] flex-1">
              <StatusDistributionBar counts={rollup.counts} />
            </div>
          </div>
        );
      },
    },
    {
      key: "workflow",
      header: "Workflow",
      width: "w-[110px]",
      cell: (exp) => {
        const rollup = workbench.experiments.find((item) => item.experiment.id === exp.id);
        return (
          <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <Workflow className="h-3.5 w-3.5" />
            {rollup?.workflowSummary.exists ? `${rollup.workflowSummary.taskCount}` : "-"}
          </span>
        );
      },
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-[160px]",
      cell: (exp) => (
        <span className="text-muted-foreground">
          {new Date(exp.updatedAt).toLocaleDateString()}
        </span>
      ),
    },
    {
      key: "action",
      header: "Action",
      width: "w-[60px]",
      align: "right",
      cell: (exp) => (
        <Button
          size="icon"
          variant="ghost"
          className="h-8 w-8 opacity-0 transition-opacity group-hover:opacity-100"
          onClick={(event) => {
            event.stopPropagation();
            setCreateRunExperimentId(exp.id);
          }}
        >
          <Play className="h-4 w-4 text-muted-foreground hover:text-foreground" />
        </Button>
      ),
    },
  ];

  const assetColumns: DataTableColumn<ApiAssetResponse>[] = [
    {
      key: "name",
      header: "Name",
      cell: (asset) => (
        <div className="flex items-center gap-2 font-medium">
          <Archive className="h-4 w-4 text-amber-500" />
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
        <span className="text-muted-foreground">
          {new Date(asset.updated_at).toLocaleDateString()}
        </span>
      ),
    },
  ];

  const createRunExperiment = createRunExperimentId
    ? snapshot.experiments.find((experiment) => experiment.id === createRunExperimentId)
    : null;

  const overviewContent = (
    <DashboardGrid>
      <DashboardCard title="Project details" className="lg:col-span-8">
        <dl className="grid gap-x-6 gap-y-3 md:grid-cols-2">
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Project ID
            </dt>
            <dd className="mt-0.5 truncate font-mono text-xs text-foreground">{project.id}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Updated
            </dt>
            <dd className="mt-0.5 truncate text-sm text-foreground">
              {new Date(project.updatedAt).toLocaleString()}
            </dd>
          </div>
          {project.summary && (
            <div className="min-w-0 md:col-span-2">
              <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Summary
              </dt>
              <dd className="mt-0.5 text-sm leading-6 text-foreground">{project.summary}</dd>
            </div>
          )}
        </dl>
      </DashboardCard>

      <DashboardCard title="Summary" className="lg:col-span-4" bodyClassName="space-y-3">
        <div>
          <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            Experiments
          </div>
          <div className="mt-1 text-2xl font-semibold tabular-nums text-foreground">
            {projectExperiments.length}
          </div>
        </div>
        <StatusDistributionBar counts={workbench.counts} />
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
          <div>
            <dt className="text-muted-foreground">Runs</dt>
            <dd className="font-semibold tabular-nums text-foreground">{projectRuns.length}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Assets</dt>
            <dd className="font-semibold tabular-nums text-foreground">{projectAssets.length}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Running</dt>
            <dd className="font-semibold tabular-nums text-info">{workbench.counts.running}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Failed</dt>
            <dd className="font-semibold tabular-nums text-destructive">
              {workbench.counts.failed}
            </dd>
          </div>
        </dl>
      </DashboardCard>

      <DashboardCard title="Run state" className="lg:col-span-6">
        {workbench.counts.total === 0 ? (
          <p className="text-xs italic text-muted-foreground">No runs recorded.</p>
        ) : (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
            <div>
              <dt className="text-muted-foreground">Succeeded</dt>
              <dd className="font-semibold tabular-nums text-success">
                {workbench.counts.succeeded}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Running</dt>
              <dd className="font-semibold tabular-nums text-info">{workbench.counts.running}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Pending</dt>
              <dd className="font-semibold tabular-nums text-warning">
                {workbench.counts.pending}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Failed</dt>
              <dd className="font-semibold tabular-nums text-destructive">
                {workbench.counts.failed}
              </dd>
            </div>
          </dl>
        )}
      </DashboardCard>

      <DashboardCard title="Assets" className="lg:col-span-6">
        {projectAssetsByKind.length === 0 ? (
          <p className="text-xs italic text-muted-foreground">No assets registered.</p>
        ) : (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
            {projectAssetsByKind.slice(0, 6).map(([kind, count]) => (
              <div key={kind} className="min-w-0">
                <dt className="truncate text-muted-foreground">{kind}</dt>
                <dd className="font-semibold tabular-nums text-foreground">{count}</dd>
              </div>
            ))}
          </dl>
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
              <DataTable
                columns={experimentColumns}
                data={projectExperiments}
                getRowKey={(exp) => exp.id}
                onRowClick={(exp) => navigateToExperiment(exp.id)}
                rowActions={experimentRowActions}
                empty={
                  <EmptyState
                    title={EMPTY_COPY.experiments.title}
                    description={EMPTY_COPY.experiments.description}
                  />
                }
              />
            ),
          },
          {
            value: "assets",
            label: "Assets",
            content: (
              <DataTable
                columns={assetColumns}
                data={projectAssets}
                getRowKey={(asset) => asset.id}
                onRowClick={(asset) => setSelection({ objectType: "asset", objectId: asset.id })}
                rowActions={assetRowActions}
                empty={
                  <EmptyState
                    title={EMPTY_COPY.assets.title}
                    description={EMPTY_COPY.assets.description}
                  />
                }
              />
            ),
          },
          {
            value: "settings",
            label: "Settings",
            content: (
              <div className="overflow-auto p-6">
                <div className="max-w-2xl space-y-5">
                  <div className="border-b border-border/70 pb-4">
                    <h3 className="text-sm font-semibold uppercase text-muted-foreground">
                      Danger Zone
                    </h3>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Deleting a project removes access to its experiment and run hierarchy from
                      this workspace view.
                    </p>
                  </div>
                  <Button variant="destructive" onClick={handleDelete} disabled={isDeleting}>
                    Delete Project
                  </Button>
                </div>
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
          onRunCreated={onRefresh}
        />
      )}
    </>
  );
};
