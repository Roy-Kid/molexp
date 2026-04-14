import { FlaskConical, Play, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { CreateRunDialog } from "@/app/components/CreateRunDialog";
import type { DataTableColumn } from "@/app/components/entity";
import {
  DataTable,
  EMPTY_COPY,
  EmptyState,
  EntityHeader,
  EntityMetric,
  KeyValueGrid,
  StatusBadge,
} from "@/app/components/entity";
import { SnapshotDiffPanel } from "@/app/renderers/SnapshotViewer";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps, RunSummary } from "@/app/types";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const ExperimentViewer = ({
  selection,
  snapshot,
  onRefresh,
}: RendererProps): JSX.Element => {
  const [isDeleting, setIsDeleting] = useState(false);
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);

  // Find the experiment in snapshot
  const experimentId = selection.objectId;
  const experiment = snapshot.experiments.find((e) => e.id === experimentId);
  const projectId = experiment?.projectId || "";

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

  if (!experiment || !projectId) {
    return <div className="p-8 text-muted-foreground">Experiment not found.</div>;
  }

  const project = snapshot.projects.find((item) => item.id === projectId);
  const workflow = snapshot.workflows.find(
    (item) => item.name === experiment.workflowFile || item.id === experiment.workflowFile,
  );

  const runColumns: DataTableColumn<RunSummary>[] = [
    {
      key: "id",
      header: "Run ID",
      width: "w-[120px]",
      cell: (run) => (
        <span className="font-mono text-xs text-muted-foreground">{run.id.substring(0, 8)}</span>
      ),
    },
    {
      key: "status",
      header: "Status",
      width: "w-[140px]",
      cell: (run) => <StatusBadge status={run.status} />,
    },
    {
      key: "summary",
      header: "Summary",
      cell: (run) => (
        <span className="block max-w-[200px] truncate text-muted-foreground" title={run.summary}>
          {run.summary || "-"}
        </span>
      ),
    },
    {
      key: "updated",
      header: "Updated",
      width: "w-[180px]",
      cell: (run) => (
        <span className="text-xs text-muted-foreground">
          {new Date(run.updatedAt).toLocaleString()}
        </span>
      ),
    },
    {
      key: "action",
      header: "Act",
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

  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
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
              className="text-muted-foreground hover:text-destructive"
              title="Delete Experiment"
            >
              <Trash2 className="h-5 w-5" />
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
      />

      <div className="flex-1 overflow-hidden flex flex-col">
        <Tabs defaultValue="runs" className="flex-1 flex flex-col">
          <div className="border-b border-border/70 bg-muted/10 px-6 py-2 md:px-8">
            <TabsList className="h-auto w-fit justify-start rounded-md bg-transparent p-0">
              <TabsTrigger value="runs" className="px-4 py-2 rounded-md font-medium text-sm">
                Runs
              </TabsTrigger>
              <TabsTrigger value="config" className="px-4 py-2 rounded-md font-medium text-sm">
                Config
              </TabsTrigger>
              <TabsTrigger value="diff" className="px-4 py-2 rounded-md font-medium text-sm">
                Diff
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="runs" className="m-0 flex flex-1 flex-col overflow-hidden p-0">
            <DataTable
              columns={runColumns}
              data={runs}
              getRowKey={(run) => run.id}
              onRowClick={(run) => navigateToRun(run.id)}
              empty={
                <EmptyState
                  title={EMPTY_COPY.runs.title}
                  description={EMPTY_COPY.runs.description}
                />
              }
            />
          </TabsContent>

          <TabsContent value="config" className="flex-1 overflow-auto p-6 md:p-8">
            <div className="max-w-3xl space-y-6">
              <div>
                <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  Relationships
                </h3>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelection({ objectType: "project", objectId: projectId })}
                  >
                    Project: {project?.name || projectId}
                  </Button>
                  {workflow && (
                    <Button
                      variant="outline"
                      size="sm"
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
              </div>

              <div>
                <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  Summary
                </h3>
                <p className="mt-3 max-w-2xl text-sm leading-6 text-foreground">
                  {experiment.summary || "No summary provided."}
                </p>
              </div>

              <div>
                <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  Metadata
                </h3>
                <div className="mt-4">
                  <KeyValueGrid
                    items={[
                      { label: "Experiment ID", value: experiment.id },
                      { label: "Project ID", value: projectId },
                      { label: "Workflow File", value: experiment.workflowFile || "-" },
                      { label: "Updated", value: new Date(experiment.updatedAt).toLocaleString() },
                    ]}
                  />
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="diff" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
            <SnapshotDiffPanel experimentRunIds={runs.map((r) => r.id)} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
