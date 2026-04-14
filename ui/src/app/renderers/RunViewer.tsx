import { FileText, Terminal } from "lucide-react";
import { useMemo } from "react";
import { EntityHeader, KeyValueGrid } from "@/app/components/entity";
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
    return <div className="p-8 text-muted-foreground">Run not found.</div>;
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
        <Tabs defaultValue="overview" className="flex-1 flex flex-col">
          <div className="border-b border-border/70 bg-muted/10 px-6 py-2 md:px-8">
            <TabsList className="h-auto w-fit justify-start rounded-md bg-transparent p-0">
              <TabsTrigger value="overview" className="px-4 py-2 rounded-md font-medium text-sm">
                Overview
              </TabsTrigger>
              <TabsTrigger value="logs" className="px-4 py-2 rounded-md font-medium text-sm">
                Logs
              </TabsTrigger>
              <TabsTrigger value="snapshot" className="px-4 py-2 rounded-md font-medium text-sm">
                Snapshot
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="overview" className="flex-1 p-0 m-0 overflow-hidden flex flex-col">
            <div className="flex-1 overflow-auto p-6 md:p-8">
              <div className="max-w-4xl space-y-6">
                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                    Relationships
                  </h3>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setSelection({ objectType: "project", objectId: run.projectId })
                      }
                    >
                      Project: {project?.name || run.projectId}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setSelection({ objectType: "experiment", objectId: run.experimentId })
                      }
                    >
                      Experiment: {experiment?.name || run.experimentId}
                    </Button>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                    Summary
                  </h3>
                  <p className="mt-3 max-w-3xl text-sm leading-6 text-foreground">
                    {run.summary || "No summary provided."}
                  </p>
                </div>

                <div>
                  <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                    Metadata
                  </h3>
                  <div className="mt-4">
                    <KeyValueGrid
                      items={displayFields.map((field) => ({
                        label: field.label,
                        value: <span className="font-mono text-xs">{field.value}</span>,
                      }))}
                    />
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="logs" className="flex-1 p-0 flex flex-col bg-slate-950 text-slate-50">
            <div className="flex items-center px-4 py-2 border-b border-slate-800 bg-slate-900 text-xs font-mono text-slate-400">
              <Terminal className="h-3 w-3 mr-2" />
              stdout/stderr
            </div>
            <div className="flex-1 p-4 font-mono text-xs overflow-auto">
              <div className="opacity-50 italic">Log streaming not implemented yet.</div>
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
