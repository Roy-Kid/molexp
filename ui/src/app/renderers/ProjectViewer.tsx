import {
  Archive,
  Ban,
  Copy,
  ExternalLink,
  FlaskConical,
  FolderKanban,
  Play,
  Terminal,
  Trash2,
  Workflow,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CreateExperimentDialog } from "@/app/components/CreateExperimentDialog";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import type { DataTableColumn, DataTableRowAction } from "@/app/components/entity";
import {
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityHeader,
  EntityMetric,
  EntityTabBar,
  EntityTabContent,
  EntityTabs,
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
  StatusBadge,
} from "@/app/components/entity";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type {
  ApiAssetResponse,
  ExperimentSummary,
  ObjectView,
  RendererProps,
  RunSummary,
} from "@/app/types";
import { Button } from "@/components/ui/button";

export const ProjectViewer = ({ selection, snapshot, onRefresh }: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [projectAssets, setProjectAssets] = useState<ApiAssetResponse[]>([]);
  const [createRunExperimentId, setCreateRunExperimentId] = useState<string | null>(null);
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);

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

  const navigateToRunView = (run: RunSummary, objectView?: ObjectView) => {
    setSelection({
      objectType: "run",
      objectId: run.id,
      objectView,
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
        icon: Trash2,
        destructive: true,
        separatorBefore: true,
        onSelect: (experiment) => {
          void handleDeleteExperiment(experiment);
        },
      },
    ];
  };

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
          <span className="font-medium text-foreground">{exp.name}</span>
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
      key: "status",
      header: "Status",
      width: "w-[160px]",
      cell: (exp) => <StatusBadge status={exp.status} />,
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-[180px]",
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

  const runColumns: DataTableColumn<RunSummary>[] = [
    {
      key: "run",
      header: "Run",
      cell: (run) => (
        <>
          <div className="font-medium text-foreground">{run.name || run.id}</div>
          <div className="font-mono text-xs text-muted-foreground">{run.id}</div>
        </>
      ),
    },
    {
      key: "experiment",
      header: "Experiment",
      width: "w-[220px]",
      cell: (run) => {
        const experiment = snapshot.experiments.find((item) => item.id === run.experimentId);
        return (
          <span className="text-muted-foreground">{experiment?.name || run.experimentId}</span>
        );
      },
    },
    {
      key: "status",
      header: "Status",
      width: "w-[140px]",
      cell: (run) => <StatusBadge status={run.status} />,
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-[180px]",
      cell: (run) => (
        <span className="text-muted-foreground">{new Date(run.updatedAt).toLocaleString()}</span>
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

  return (
    <>
      <div className="flex h-full flex-col bg-background">
        <EntityHeader
          breadcrumbs={breadcrumbs}
          canNavigateUp={canNavigateUp}
          onNavigateUp={navigateUp}
          icon={FolderKanban}
          title={project.name}
          status={project.status}
          subtitle={project.summary || undefined}
          actions={
            <>
              <CreateExperimentDialog projectId={projectId} onExperimentCreated={onRefresh} />
              <Button
                variant="ghost"
                size="icon"
                onClick={handleDelete}
                disabled={isDeleting}
                className="text-muted-foreground hover:text-destructive"
                title="Delete Project"
              >
                <Trash2 className="h-5 w-5" />
              </Button>
            </>
          }
          metrics={
            <>
              <EntityMetric label="Experiments" value={projectExperiments.length} />
              <EntityMetric label="Runs" value={projectRuns.length} />
              <EntityMetric label="Assets" value={projectAssets.length} />
            </>
          }
        />

        <div className="flex-1 flex flex-col overflow-hidden">
          <EntityTabs defaultValue="overview">
            <EntityTabBar
              tabs={[
                { value: "overview", label: "Overview" },
                { value: "experiments", label: "Experiments" },
                { value: "runs", label: "Runs" },
                { value: "assets", label: "Assets" },
                { value: "settings", label: "Settings" },
              ]}
            />

            <EntityTabContent value="overview">
              <OverviewPage
                aside={
                  <>
                    <OverviewSection title="Inventory">
                      <OverviewHighlightGrid>
                        <OverviewHighlight label="Experiments" value={projectExperiments.length} />
                        <OverviewHighlight label="Runs" value={projectRuns.length} />
                        <OverviewHighlight label="Assets" value={projectAssets.length} />
                        <OverviewHighlight
                          label="Updated"
                          value={new Date(project.updatedAt).toLocaleString()}
                        />
                      </OverviewHighlightGrid>
                    </OverviewSection>

                    <OverviewSection title="Status">
                      <OverviewHighlightGrid>
                        <OverviewHighlight label="Workspace state" value={project.status} />
                      </OverviewHighlightGrid>
                    </OverviewSection>
                  </>
                }
              >
                <OverviewSection title="Summary">
                  <p className="max-w-3xl text-sm leading-6 text-foreground">
                    {project.summary || (
                      <span className="text-muted-foreground">No summary provided.</span>
                    )}
                  </p>
                </OverviewSection>

                <OverviewSection title="Identity">
                  <KeyValueGrid
                    items={[
                      { label: "Project ID", value: project.id },
                      { label: "Name", value: project.name },
                      { label: "Status", value: project.status },
                      { label: "Updated", value: new Date(project.updatedAt).toLocaleString() },
                    ]}
                  />
                </OverviewSection>
              </OverviewPage>
            </EntityTabContent>

            <EntityTabContent value="experiments">
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
            </EntityTabContent>

            <EntityTabContent value="runs">
              <DataTable
                columns={runColumns}
                data={projectRuns}
                getRowKey={(run) => run.id}
                onRowClick={(run) => navigateToRunView(run)}
                rowActions={runRowActions}
                empty={
                  <EmptyState
                    title={EMPTY_COPY.projectRuns.title}
                    description={EMPTY_COPY.projectRuns.description}
                  />
                }
              />
            </EntityTabContent>

            <EntityTabContent value="assets">
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
            </EntityTabContent>

            <EntityTabContent value="settings" className="overflow-auto p-6">
              <div className="max-w-2xl space-y-5">
                <div className="border-b border-border/70 pb-4">
                  <h3 className="text-sm font-semibold uppercase text-muted-foreground">
                    Danger Zone
                  </h3>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Deleting a project removes access to its experiment and run hierarchy from this
                    workspace view.
                  </p>
                </div>
                <Button variant="destructive" onClick={handleDelete} disabled={isDeleting}>
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete Project
                </Button>
              </div>
            </EntityTabContent>
          </EntityTabs>
        </div>
      </div>
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
