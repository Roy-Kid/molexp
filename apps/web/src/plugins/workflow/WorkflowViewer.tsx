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
import { WorkflowGraphViewer } from "@/plugins/workflow/WorkflowGraphViewer";
import { WorkflowSourceViewer } from "@/plugins/workflow/WorkflowSourceViewer";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { WorkbenchAction } from "@/components/workbench";
import { formatDateTime } from "@/lib/datetime";

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
              <OverviewHighlight label="Updated" value={formatDateTime(workflow.updatedAt)} />
            </OverviewHighlightGrid>
          </OverviewSection>

          <OverviewSection title="Relationships">
            <div className="flex flex-wrap gap-2">
              <WorkbenchAction
                kind="secondary"
                size="compact"
                className="h-control-compact px-2 text-label"
                onClick={() =>
                  setSelection({ objectType: "project", objectId: workflow.projectId })
                }
              >
                Project: {project?.name || workflow.projectId}
              </WorkbenchAction>
              <WorkbenchAction
                kind="secondary"
                size="compact"
                className="h-control-compact px-2 text-label"
                onClick={() =>
                  setSelection({ objectType: "experiment", objectId: workflow.experimentId })
                }
              >
                Experiment: {experiment?.name || workflow.experimentId}
              </WorkbenchAction>
            </div>
          </OverviewSection>
        </>
      }
    >
      <OverviewSection title="Summary">
        <p className="max-w-3xl text-body-lg leading-6 text-foreground">
          {workflow.summary || <span className="text-muted-foreground">No summary provided.</span>}
        </p>
      </OverviewSection>

      <OverviewSection title="Metadata">
        <KeyValueGrid
          items={[
            {
              label: "Workflow ID",
              value: <span className="font-mono text-label">{workflow.id}</span>,
            },
            {
              label: "Project ID",
              value: <span className="font-mono text-label">{workflow.projectId}</span>,
            },
            {
              label: "Experiment ID",
              value: <span className="font-mono text-label">{workflow.experimentId}</span>,
            },
            {
              label: "Last Updated",
              value: <span title={workflow.updatedAt}>{formatDateTime(workflow.updatedAt)}</span>,
            },
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
