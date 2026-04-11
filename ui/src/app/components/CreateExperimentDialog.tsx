import { Plus } from "lucide-react";
import { useState } from "react";
import { workspaceApi } from "@/app/state/api";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

interface CreateExperimentDialogProps {
  projectId: string;
  onExperimentCreated: () => void;
}

export function CreateExperimentDialog({
  projectId,
  onExperimentCreated,
}: CreateExperimentDialogProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [id, setId] = useState("");
  const [workflow, setWorkflow] = useState("");
  const [description, setDescription] = useState("");
  const [parameterSpace, setParameterSpace] = useState("{}");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      // Auto-generate ID if empty
      const experimentId = id || name.toLowerCase().replace(/[^a-z0-9-]/g, "-");

      await workspaceApi.createExperiment(projectId, {
        id: experimentId,
        name,
        workflow_source: workflow,
        description,
        parameter_space: JSON.parse(parameterSpace),
      });

      setOpen(false);
      setName("");
      setId("");
      setWorkflow("");
      setDescription("");
      setParameterSpace("{}");
      onExperimentCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create experiment");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="h-7 gap-1">
          <Plus className="h-3.5 w-3.5" />
          <span className="sr-only sm:not-sr-only sm:whitespace-nowrap">New Experiment</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[550px]">
        <DialogHeader>
          <DialogTitle>Create Experiment</DialogTitle>
          <DialogDescription>
            Defines a computable experiment within project {projectId}.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="exp-name" className="text-right">
                Name
              </Label>
              <Input
                id="exp-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="col-span-3"
                required
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="exp-workflow" className="text-right">
                Workflow File
              </Label>
              <Input
                id="exp-workflow"
                value={workflow}
                onChange={(e) => setWorkflow(e.target.value)}
                placeholder="path/to/workflow.yaml"
                className="col-span-3"
                required
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="exp-params" className="text-right">
                Parameters (JSON)
              </Label>
              <Textarea
                id="exp-params"
                value={parameterSpace}
                onChange={(e) => setParameterSpace(e.target.value)}
                className="col-span-3 font-mono text-xs"
                rows={4}
              />
            </div>
            {error && <div className="text-sm text-red-500 col-span-4 text-center">{error}</div>}
          </div>
          <DialogFooter>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? "Creating..." : "Create Experiment"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
