import { Archive, FlaskConical, FolderKanban, Play, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CreateExperimentDialog } from "@/app/components/CreateExperimentDialog";
import type { DataTableColumn } from "@/app/components/entity";
import {
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityHeader,
  EntityMetric,
  StatusBadge,
} from "@/app/components/entity";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ApiAssetResponse, ExperimentSummary, RendererProps, RunSummary } from "@/app/types";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const ProjectViewer = ({ selection, snapshot, onRefresh }: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [projectAssets, setProjectAssets] = useState<ApiAssetResponse[]>([]);
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
      cell: () => (
        <Button
          size="icon"
          variant="ghost"
          className="h-8 w-8 opacity-0 transition-opacity group-hover:opacity-100"
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
      header: "Asset Name",
      cell: (asset) => (
        <div className="flex items-center gap-2 font-medium">
          <Archive className="h-4 w-4 text-amber-500" />
          {asset.assetId}
        </div>
      ),
    },
    {
      key: "type",
      header: "Type",
      width: "w-[150px]",
      cell: (asset) => <span className="text-muted-foreground">{asset.format}</span>,
    },
    {
      key: "size",
      header: "Size",
      width: "w-[120px]",
      cell: (asset) => <span className="font-mono text-xs">{asset.size} B</span>,
    },
    {
      key: "created",
      header: "Created",
      width: "w-[180px]",
      cell: (asset) => (
        <span className="text-muted-foreground">
          {new Date(asset.created).toLocaleDateString()}
        </span>
      ),
    },
  ];

  return (
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
        <Tabs defaultValue="experiments" className="flex-1 flex flex-col">
          <div className="border-b border-border/70 bg-muted/10 px-6 py-2 md:px-8">
            <TabsList className="h-auto w-fit justify-start rounded-md bg-transparent p-0">
              <TabsTrigger value="experiments" className="rounded-md px-4 py-2 text-sm font-medium">
                Experiments
              </TabsTrigger>
              <TabsTrigger value="runs" className="rounded-md px-4 py-2 text-sm font-medium">
                Runs
              </TabsTrigger>
              <TabsTrigger value="assets" className="rounded-md px-4 py-2 text-sm font-medium">
                Assets
              </TabsTrigger>
              <TabsTrigger value="settings" className="rounded-md px-4 py-2 text-sm font-medium">
                Settings
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="experiments" className="m-0 flex flex-1 flex-col overflow-hidden p-0">
            <DataTable
              columns={experimentColumns}
              data={projectExperiments}
              getRowKey={(exp) => exp.id}
              onRowClick={(exp) => navigateToExperiment(exp.id)}
              empty={
                <EmptyState
                  title={EMPTY_COPY.experiments.title}
                  description={EMPTY_COPY.experiments.description}
                />
              }
            />
          </TabsContent>

          <TabsContent value="runs" className="m-0 flex flex-1 flex-col overflow-hidden p-0">
            <DataTable
              columns={runColumns}
              data={projectRuns}
              getRowKey={(run) => run.id}
              onRowClick={(run) => setSelection({ objectType: "run", objectId: run.id })}
              empty={
                <EmptyState
                  title={EMPTY_COPY.projectRuns.title}
                  description={EMPTY_COPY.projectRuns.description}
                />
              }
            />
          </TabsContent>

          <TabsContent value="assets" className="m-0 flex flex-1 flex-col overflow-hidden p-0">
            <DataTable
              columns={assetColumns}
              data={projectAssets}
              getRowKey={(asset) => asset.id}
              empty={
                <EmptyState
                  title={EMPTY_COPY.assets.title}
                  description={EMPTY_COPY.assets.description}
                />
              }
            />
          </TabsContent>

          <TabsContent value="settings" className="flex-1 p-6">
            <div className="max-w-2xl space-y-5">
              <div className="border-b border-border/70 pb-4">
                <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
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
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
