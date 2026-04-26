import {
  Archive,
  Ban,
  Copy,
  ExternalLink,
  FileQuestion,
  FlaskConical,
  Terminal,
  Trash2,
} from "lucide-react";
import { useMemo, useState } from "react";
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
import { SnapshotDiffPanel } from "@/app/renderers/SnapshotViewer";
import { workspaceApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { ObjectView, RendererProps, RunSummary } from "@/app/types";
import { Button } from "@/components/ui/button";

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

  const navigateToRunView = (run: RunSummary, objectView?: ObjectView) => {
    setSelection({
      objectType: "run",
      objectId: run.id,
      objectView,
    });
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
        <EntityTabs defaultValue="runs">
          <EntityTabBar
            tabs={[
              { value: "runs", label: "Runs" },
              { value: "overview", label: "Overview" },
              { value: "diff", label: "Diff" },
            ]}
          />

          <EntityTabContent value="runs">
            <DataTable
              columns={runColumns}
              data={runs}
              getRowKey={(run) => run.id}
              onRowClick={(run) => navigateToRun(run.id)}
              rowActions={runRowActions}
              empty={
                <EmptyState
                  title={EMPTY_COPY.runs.title}
                  description={EMPTY_COPY.runs.description}
                />
              }
            />
          </EntityTabContent>

          <EntityTabContent value="overview">
            <OverviewPage
              aside={
                <>
                  <OverviewSection title="Run Summary">
                    <OverviewHighlightGrid>
                      <OverviewHighlight label="Total" value={stats.total} detail="runs" />
                      <OverviewHighlight label="Succeeded" value={stats.succeeded} />
                      <OverviewHighlight label="Failed" value={stats.failed} />
                      <OverviewHighlight label="Running" value={stats.running} />
                    </OverviewHighlightGrid>
                  </OverviewSection>

                  <OverviewSection title="Relationships">
                    <div className="flex flex-wrap gap-1.5">
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
                  </OverviewSection>
                </>
              }
            >
              <OverviewSection title="Summary">
                <p className="max-w-3xl text-sm leading-6 text-foreground">
                  {experiment.summary || (
                    <span className="text-muted-foreground">No summary provided.</span>
                  )}
                </p>
              </OverviewSection>

              <OverviewSection title="Metadata">
                <KeyValueGrid
                  items={[
                    { label: "Experiment ID", value: experiment.id },
                    { label: "Project ID", value: projectId },
                    { label: "Workflow File", value: experiment.workflowFile || "-" },
                    { label: "Updated", value: new Date(experiment.updatedAt).toLocaleString() },
                  ]}
                />
              </OverviewSection>
            </OverviewPage>
          </EntityTabContent>

          <EntityTabContent value="diff">
            <SnapshotDiffPanel experimentRunIds={runs.map((r) => r.id)} />
          </EntityTabContent>
        </EntityTabs>
      </div>
    </div>
  );
};
