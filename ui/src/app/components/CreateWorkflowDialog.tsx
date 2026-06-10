import type { ReactNode } from "react";
import { useState } from "react";
import { workflowApi, workspaceApi } from "@/app/state/api";
import type { ProjectSummary } from "@/app/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/** Wire IR seeded onto a brand-new workflow so the canvas opens empty. */
const EMPTY_WORKFLOW_DOCUMENT = { task_configs: [], links: [] };

interface CreateWorkflowDialogProps {
  projects: ProjectSummary[];
  /** Called with the new experiment id after the empty document is seeded. */
  onCreated: (experimentId: string) => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  trigger?: ReactNode;
}

/**
 * Two-field dialog (Project + Name) that creates an experiment, seeds it with
 * an empty workflow document, and hands the new experiment id to the caller —
 * the WorkflowsPage entry point for drafting a workflow from scratch.
 */
export function CreateWorkflowDialog({
  projects,
  onCreated,
  open: controlledOpen,
  onOpenChange,
  trigger,
}: CreateWorkflowDialogProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const [projectId, setProjectId] = useState("");
  const [name, setName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = onOpenChange ?? setUncontrolledOpen;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!projectId) return;
    setIsLoading(true);
    setError(null);

    try {
      const experiment = await workspaceApi.createExperiment(projectId, {
        name,
        parameter_space: {},
      });
      await workflowApi.save(projectId, experiment.id, EMPTY_WORKFLOW_DOCUMENT);

      setOpen(false);
      setProjectId("");
      setName("");
      onCreated(experiment.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create workflow");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      {trigger ? <DialogTrigger asChild>{trigger}</DialogTrigger> : null}
      <DialogContent className="sm:max-w-[425px]" aria-describedby={undefined}>
        <DialogHeader>
          <DialogTitle>New workflow</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="wf-project" className="text-right">
                Project
              </Label>
              <Select value={projectId} onValueChange={setProjectId}>
                <SelectTrigger id="wf-project" className="col-span-3">
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  {projects.map((project) => (
                    <SelectItem key={project.id} value={project.id}>
                      {project.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="wf-name" className="text-right">
                Name
              </Label>
              <Input
                id="wf-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="col-span-3"
                required
              />
            </div>
            {error && <div className="text-sm text-red-500 text-center">{error}</div>}
          </div>
          <DialogFooter>
            <Button type="submit" disabled={isLoading || !projectId}>
              {isLoading ? "Creating..." : "Create"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
