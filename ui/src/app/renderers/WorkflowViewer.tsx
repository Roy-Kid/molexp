import { Workflow } from "lucide-react";
import {
  EntityHeader,
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
} from "@/app/components/entity";
import { TabbedViewer } from "@/app/renderers/TabbedViewer";
import { WorkflowGraphViewer } from "@/app/renderers/WorkflowGraphViewer";
import { WorkflowSourceViewer } from "@/app/renderers/WorkflowSourceViewer";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";

const WorkflowOverview = ({ selection, snapshot }: RendererProps): JSX.Element => {
  const workflowId = selection.objectId;
  const workflow = snapshot.workflows.find((w) => w.id === workflowId);
  const { setSelection, breadcrumbs, canNavigateUp, navigateUp } = useNavigationState(snapshot);

  if (!workflow) {
    return <div className="p-8 text-muted-foreground">Workflow not found.</div>;
  }

  const project = snapshot.projects.find((item) => item.id === workflow.projectId);
  const experiment = snapshot.experiments.find((item) => item.id === workflow.experimentId);
  const nodeCount = workflow.graph?.nodes.length ?? 0;
  const edgeCount = workflow.graph?.edges.length ?? 0;

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

      <OverviewPage
        aside={
          <>
            <OverviewSection title="Highlights">
              <OverviewHighlightGrid>
                <OverviewHighlight label="Status" value={workflow.status} />
                <OverviewHighlight label="Nodes" value={nodeCount} />
                <OverviewHighlight label="Edges" value={edgeCount} />
                <OverviewHighlight
                  label="Updated"
                  value={new Date(workflow.updatedAt).toLocaleString()}
                />
              </OverviewHighlightGrid>
            </OverviewSection>

            <OverviewSection title="Relationships">
              <div className="flex flex-wrap gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() =>
                    setSelection({ objectType: "project", objectId: workflow.projectId })
                  }
                >
                  Project: {project?.name || workflow.projectId}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() =>
                    setSelection({ objectType: "experiment", objectId: workflow.experimentId })
                  }
                >
                  Experiment: {experiment?.name || workflow.experimentId}
                </Button>
              </div>
            </OverviewSection>
          </>
        }
      >
        <OverviewSection title="Summary">
          <p className="max-w-3xl text-sm leading-6 text-foreground">
            {workflow.summary || (
              <span className="text-muted-foreground">No summary provided.</span>
            )}
          </p>
        </OverviewSection>

        <OverviewSection title="Metadata">
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
        </OverviewSection>
      </OverviewPage>
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
