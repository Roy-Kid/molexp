import { Plus, Workflow as WorkflowIcon } from "lucide-react";
import type { JSX } from "react";
import { useState } from "react";
import { CreateWorkflowDialog } from "@/app/components/CreateWorkflowDialog";
import { EmptyState, EntityHeader } from "@/app/components/entity";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { WorkspaceSnapshot } from "@/app/types";
import { RunStatusBadge, WorkbenchAction } from "@/components/workbench";

interface WorkflowsPageProps {
  snapshot: WorkspaceSnapshot;
  onRefresh: () => void;
}

/**
 * WorkflowsPage — the landing page for the ``/workflows`` section. Lists every
 * workflow (one per experiment); clicking a row opens the full graph viewer.
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
    <WorkbenchAction kind="primary" size="compact" onClick={() => setCreateOpen(true)}>
      <Plus className="h-3.5 w-3.5" />
      New workflow
    </WorkbenchAction>
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
      <div className="flex-1 overflow-auto">
        {workflows.length === 0 ? (
          <div className="flex h-full items-center justify-center p-4">
            <EmptyState title="No workflows yet." action={newWorkflowButton} />
          </div>
        ) : (
          <ul className="divide-y divide-border border-t border-border">
            {workflows.map((workflow) => {
              const experiment = snapshot.experiments.find(
                (item) => item.id === workflow.experimentId,
              );
              const nodeCount = workflow.graph?.task_configs.length ?? 0;
              const edgeCount = workflow.graph?.links.length ?? 0;
              return (
                <li key={workflow.id}>
                  <button
                    type="button"
                    className="flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-interactive/50"
                    onClick={() =>
                      setSelection({
                        objectType: "workflow",
                        objectId: workflow.id,
                        workflowId: workflow.id,
                      })
                    }
                  >
                    <WorkflowIcon className="mt-1 h-4 w-4 flex-none text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate text-body font-medium text-foreground">
                          {workflow.name}
                        </span>
                        <RunStatusBadge status={workflow.status} size="sm" />
                      </div>
                      <p className="mt-1 text-label text-muted-foreground">
                        {nodeCount} tasks · {edgeCount} dependencies
                        {experiment ? ` · ${experiment.name}` : ""}
                      </p>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
};
