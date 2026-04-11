import { Badge } from "@/components/ui/badge";
import { Workflow } from "lucide-react";
import type { RendererProps } from "@/app/types";
import { TabbedViewer } from "@/app/renderers/TabbedViewer";
import { WorkflowGraphViewer } from "@/app/renderers/WorkflowGraphViewer";
import { WorkflowSourceViewer } from "@/app/renderers/WorkflowSourceViewer";

const WorkflowOverview = ({
  selection,
  snapshot,
}: RendererProps): JSX.Element => {
  const workflowId = selection.objectId;
  const workflow = snapshot.workflows.find((w) => w.id === workflowId);

  if (!workflow) {
    return <div className="p-8 text-muted-foreground">Workflow not found.</div>;
  }

  return (
    <div className="flex h-full flex-col bg-background overflow-y-auto">
      <div className="flex flex-col gap-6 px-8 py-8 border-b bg-background">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Workflow className="h-6 w-6 text-blue-600" />
              </div>
              <h1 className="text-3xl font-bold tracking-tight text-foreground">{workflow.name}</h1>
            </div>
            <div className="flex items-center gap-2 pl-[3.25rem]">
              <Badge variant="outline" className="font-mono text-xs text-muted-foreground">
                {workflowId}
              </Badge>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 p-8 space-y-6">
        <div className="space-y-2">
             <h3 className="text-lg font-semibold">Summary</h3>
             <div className="rounded-md border p-4 bg-muted/20">
                <p className="text-sm">
                    {workflow.summary || "No summary available."}
                </p>
             </div>
        </div>

        <div className="space-y-2">
            <h3 className="text-lg font-semibold">Metadata</h3>
             <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-md border bg-card">
                    <p className="text-xs text-muted-foreground uppercase">Project ID</p>
                    <p className="font-mono text-sm">{workflow.projectId}</p>
                </div>
                <div className="p-4 rounded-md border bg-card">
                    <p className="text-xs text-muted-foreground uppercase">Experiment ID</p>
                    <p className="font-mono text-sm">{workflow.experimentId}</p>
                </div>
                <div className="p-4 rounded-md border bg-card">
                    <p className="text-xs text-muted-foreground uppercase">Last Updated</p>
                    <p className="text-sm">{new Date(workflow.updatedAt).toLocaleString()}</p>
                </div>
                <div className="p-4 rounded-md border bg-card">
                    <p className="text-xs text-muted-foreground uppercase">Status</p>
                    <p className="text-sm capitalize">{workflow.status}</p>
                </div>
             </div>
        </div>
      </div>
    </div>
  );
};

export const WorkflowViewer = (props: RendererProps): JSX.Element => {
  return (
    <TabbedViewer
      {...props}
      defaultTab="graph"
      tabs={[
        { id: "graph", label: "Graph", component: WorkflowGraphViewer },
        { id: "source", label: "Source", component: WorkflowSourceViewer },
        { id: "overview", label: "Overview", component: WorkflowOverview },
      ]}
    />
  );
};
