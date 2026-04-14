import { Workflow } from "lucide-react";
import { EntityHeader, KeyValueGrid } from "@/app/components/entity";
import { TabbedViewer } from "@/app/renderers/TabbedViewer";
import { WorkflowGraphViewer } from "@/app/renderers/WorkflowGraphViewer";
import { WorkflowSourceViewer } from "@/app/renderers/WorkflowSourceViewer";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";

const WorkflowOverview = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const workflowId = selection.objectId;
  const workflow = snapshot.workflows.find((w) => w.id === workflowId);
  const { breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);

  if (!workflow) {
    return <div className="p-8 text-muted-foreground">Workflow not found.</div>;
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-background">
      <EntityHeader
        breadcrumbs={breadcrumbs}
        canNavigateUp={canNavigateUp}
        onNavigateUp={navigateUp}
        icon={Workflow}
        title={workflow.name}
        status={workflow.status}
        subtitle={workflow.summary || undefined}
      />

      <div className="flex-1 overflow-auto p-6 md:p-8">
        <div className="max-w-3xl space-y-6">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Summary
            </h3>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-foreground">
              {workflow.summary || "No summary provided."}
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Metadata
            </h3>
            <div className="mt-4">
              <KeyValueGrid
                items={[
                  {
                    label: "Workflow ID",
                    value: <span className="font-mono text-xs">{workflow.id}</span>,
                  },
                  {
                    label: "Project ID",
                    value: <span className="font-mono text-xs">{workflow.projectId}</span>,
                  },
                  {
                    label: "Experiment ID",
                    value: <span className="font-mono text-xs">{workflow.experimentId}</span>,
                  },
                  { label: "Last Updated", value: new Date(workflow.updatedAt).toLocaleString() },
                ]}
              />
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
