import { FileQuestion, FlaskConical, Play, Trash2 } from "lucide-react";
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
      header: "",
      width: "w-[40px]",
      align: "right",
      cell: () => (
        <Button
          size="icon"
          variant="ghost"
          aria-label="Run again"
          className="h-6 w-6 text-muted-foreground opacity-60 transition-opacity group-hover:opacity-100 hover:text-foreground"
        >
          <Play className="h-3.5 w-3.5" />
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
      />

      <div className="flex-1 overflow-hidden flex flex-col">
        <Tabs defaultValue="runs" className="flex flex-1 flex-col">
          <div className="border-b border-border/70 bg-muted/20 px-4">
            <TabsList className="h-auto w-fit justify-start gap-0 rounded-none bg-transparent p-0">
              <TabsTrigger value="runs" className="rounded-none px-3 py-1.5 text-xs">
                Runs
              </TabsTrigger>
              <TabsTrigger value="config" className="rounded-none px-3 py-1.5 text-xs">
                Config
              </TabsTrigger>
              <TabsTrigger value="diff" className="rounded-none px-3 py-1.5 text-xs">
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

          <TabsContent value="config" className="flex-1 space-y-4 overflow-auto p-4">
            <section>
              <h3 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Relationships
              </h3>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
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
            </section>

            <section>
              <h3 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Summary
              </h3>
              <p className="mt-1.5 text-sm leading-5 text-foreground">
                {experiment.summary || (
                  <span className="text-muted-foreground">No summary provided.</span>
                )}
              </p>
            </section>

            <section>
              <h3 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Metadata
              </h3>
              <div className="mt-2">
                <KeyValueGrid
                  items={[
                    { label: "Experiment ID", value: experiment.id },
                    { label: "Project ID", value: projectId },
                    { label: "Workflow File", value: experiment.workflowFile || "-" },
                    { label: "Updated", value: new Date(experiment.updatedAt).toLocaleString() },
                  ]}
                />
              </div>
            </section>
          </TabsContent>

          <TabsContent value="diff" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
            <SnapshotDiffPanel experimentRunIds={runs.map((r) => r.id)} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
