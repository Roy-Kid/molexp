import { Plus, Workflow as WorkflowIcon } from "lucide-react";
import type { JSX } from "react";
import { useState } from "react";
import { CreateWorkflowDialog } from "@/app/components/CreateWorkflowDialog";
import { EmptyState, EntityHeader, StatusBadge } from "@/app/components/entity";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { WorkspaceSnapshot } from "@/app/types";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface WorkflowsPageProps {
  snapshot: WorkspaceSnapshot;
  onRefresh: () => void;
}

/**
 * WorkflowsPage — the landing page for the ``/workflows`` section. Lists every
 * workflow (one per experiment) as a card showing its task/dependency counts;
 * clicking a card opens the full workflow graph viewer. "New workflow" creates
 * an experiment seeded with an empty document and jumps straight to its canvas.
 */
export const WorkflowsPage = ({ snapshot, onRefresh }: WorkflowsPageProps): JSX.Element => {
  const { setSelection } = useNavigationState(snapshot);
  const workflows = snapshot.workflows;
  const [createOpen, setCreateOpen] = useState(false);

  const handleCreated = (experimentId: string): void => {
    onRefresh();
    const workflowId = `workflow:${experimentId}`;
    setSelection({ objectType: "workflow", objectId: workflowId, workflowId });
  };

  const newWorkflowButton = (
    <Button size="sm" className="h-7 gap-1" onClick={() => setCreateOpen(true)}>
      <Plus className="h-3.5 w-3.5" />
      New workflow
    </Button>
  );

  return (
    <div className="flex h-full flex-col">
      <EntityHeader
        icon={WorkflowIcon}
        title="Workflows"
        subtitle="Workflow definitions across the workspace — open one to inspect its task graph."
        actions={newWorkflowButton}
      />
      <CreateWorkflowDialog
        projects={snapshot.projects}
        onCreated={handleCreated}
        open={createOpen}
        onOpenChange={setCreateOpen}
      />
      <div className="flex-1 overflow-auto p-4">
        {workflows.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <EmptyState title="No workflows yet." action={newWorkflowButton} />
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {workflows.map((workflow) => {
              const experiment = snapshot.experiments.find(
                (item) => item.id === workflow.experimentId,
              );
              const nodeCount = workflow.graph?.task_configs.length ?? 0;
              const edgeCount = workflow.graph?.links.length ?? 0;
              return (
                <Card
                  key={workflow.id}
                  className="cursor-pointer p-4 transition-colors hover:border-border hover:bg-muted/30"
                  onClick={() =>
                    setSelection({
                      objectType: "workflow",
                      objectId: workflow.id,
                      workflowId: workflow.id,
                    })
                  }
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-2">
                      <WorkflowIcon className="h-4 w-4 flex-none text-muted-foreground" />
                      <span className="truncate text-sm font-semibold text-foreground">
                        {workflow.name}
                      </span>
                    </div>
                    <StatusBadge status={workflow.status} />
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    {nodeCount} tasks · {edgeCount} dependencies
                  </p>
                  {experiment && (
                    <p className="mt-1 truncate text-[11px] text-muted-foreground">
                      experiment: {experiment.name}
                    </p>
                  )}
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
