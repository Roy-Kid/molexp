import { FileQuestion, FileText, Terminal } from "lucide-react";
import { useMemo } from "react";
import { EmptyState, EntityHeader, KeyValueGrid } from "@/app/components/entity";
import { buildMetadataFields } from "@/app/renderers/metadata";
import { RunSnapshotPanel } from "@/app/renderers/SnapshotViewer";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const RunViewer = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);
  const fields = buildMetadataFields(selection, snapshot);

  const run = useMemo(() => {
    return snapshot.runs.find((r) => r.id === selection.objectId);
  }, [snapshot.runs, selection.objectId]);

  if (!run) {
    return (
      <div className="flex h-full items-center justify-center bg-background">
        <EmptyState
          icon={<FileQuestion className="h-6 w-6" />}
          title="Run not found"
          description="It may have been deleted or not yet synced."
        />
      </div>
    );
  }

  const project = snapshot.projects.find((item) => item.id === run.projectId);
  const experiment = snapshot.experiments.find((item) => item.id === run.experimentId);

  // Filter interesting fields to show in the table
  const displayFields = fields.filter(
    (f) => !["Run", "Status", "Summary", "Project", "Experiment"].includes(f.label),
  );

  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
        breadcrumbs={breadcrumbs}
        canNavigateUp={canNavigateUp}
        onNavigateUp={navigateUp}
        icon={FileText}
        title={run.name}
        status={run.status}
        subtitle={run.summary || undefined}
      />

      <div className="flex-1 overflow-hidden flex flex-col">
        <Tabs defaultValue="overview" className="flex flex-1 flex-col">
          <div className="border-b border-border/70 bg-muted/20 px-4">
            <TabsList className="h-auto w-fit justify-start gap-0 rounded-none bg-transparent p-0">
              <TabsTrigger value="overview" className="rounded-none px-3 py-1.5 text-xs">
                Overview
              </TabsTrigger>
              <TabsTrigger value="logs" className="rounded-none px-3 py-1.5 text-xs">
                Logs
              </TabsTrigger>
              <TabsTrigger value="snapshot" className="rounded-none px-3 py-1.5 text-xs">
                Snapshot
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="overview" className="m-0 flex flex-1 flex-col overflow-hidden p-0">
            <div className="flex-1 space-y-4 overflow-auto p-4">
              <section>
                <h3 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Relationships
                </h3>
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    onClick={() =>
                      setSelection({ objectType: "project", objectId: run.projectId })
                    }
                  >
                    Project: {project?.name || run.projectId}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    onClick={() =>
                      setSelection({ objectType: "experiment", objectId: run.experimentId })
                    }
                  >
                    Experiment: {experiment?.name || run.experimentId}
                  </Button>
                </div>
              </section>

              <section>
                <h3 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Summary
                </h3>
                <p className="mt-1.5 text-sm leading-5 text-foreground">
                  {run.summary || (
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
                    items={displayFields.map((field) => ({
                      label: field.label,
                      value: <span className="font-mono text-xs">{field.value}</span>,
                    }))}
                  />
                </div>
              </section>
            </div>
          </TabsContent>

          <TabsContent value="logs" className="m-0 flex flex-1 flex-col overflow-hidden bg-zinc-950 p-0 text-zinc-50 dark:bg-black">
            <div className="flex items-center gap-2 border-b border-zinc-800 bg-zinc-900 px-3 py-1 font-mono text-[11px] text-zinc-400">
              <Terminal className="h-3 w-3" />
              stdout/stderr
            </div>
            <div className="flex-1 overflow-auto p-3 font-mono text-xs">
              <div className="italic opacity-60">Log streaming not implemented yet.</div>
            </div>
          </TabsContent>

          <TabsContent value="snapshot" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
            <RunSnapshotPanel runId={run.id} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
