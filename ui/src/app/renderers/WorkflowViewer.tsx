import { Workflow } from "lucide-react";
import { useState } from "react";
import {
  EntityPage,
  KeyValueGrid,
  OverviewHighlight,
  OverviewHighlightGrid,
  OverviewPage,
  OverviewSection,
} from "@/app/components/entity";
import { WorkflowGraphViewer } from "@/app/renderers/WorkflowGraphViewer";
import { WorkflowSourceViewer } from "@/app/renderers/WorkflowSourceViewer";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";

const WorkflowOverviewBody = ({ selection, snapshot }: RendererProps): JSX.Element | null => {
  const workflow = snapshot.workflows.find((w) => w.id === selection.objectId);
  const { setSelection } = useNavigationState(snapshot);
  if (!workflow) return null;

  const project = snapshot.projects.find((item) => item.id === workflow.projectId);
  const experiment = snapshot.experiments.find((item) => item.id === workflow.experimentId);
  const nodeCount = workflow.graph?.task_configs.length ?? 0;
  const edgeCount = workflow.graph?.links.length ?? 0;

  return (
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
          {workflow.summary || <span className="text-muted-foreground">No summary provided.</span>}
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
  );
};

export const WorkflowViewer = (props: RendererProps): JSX.Element => {
  const { selection, snapshot } = props;
  const workflow = snapshot.workflows.find((w) => w.id === selection.objectId);
  const [activeTab, setActiveTab] = useState("graph");

  if (!workflow) {
    return <div className="p-8 text-muted-foreground">Workflow not found.</div>;
  }

  return (
    <EntityPage
      icon={Workflow}
      title={workflow.name}
      status={workflow.status}
      subtitle={workflow.summary || undefined}
      activeTab={activeTab}
      onActiveTabChange={setActiveTab}
      tabs={[
        {
          value: "graph",
          label: "Graph",
          content: activeTab === "graph" ? <WorkflowGraphViewer {...props} /> : null,
        },
        { value: "overview", label: "Overview", content: <WorkflowOverviewBody {...props} /> },
        { value: "source", label: "Source", content: <WorkflowSourceViewer {...props} /> },
      ]}
    />
  );
};
