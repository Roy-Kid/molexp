import { Play } from "lucide-react";
import type { ReactNode } from "react";
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

interface CreateRunDialogProps {
  projectId: string;
  experimentId: string;
  workflowFile: string;
  onRunCreated: () => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  trigger?: ReactNode;
}

export function CreateRunDialog({
  projectId,
  experimentId,
  workflowFile,
  onRunCreated,
  open: controlledOpen,
  onOpenChange,
  trigger,
}: CreateRunDialogProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const [parameters, setParameters] = useState("{}");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = onOpenChange ?? setUncontrolledOpen;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      await workspaceApi.createRun(projectId, experimentId, {
        parameters: JSON.parse(parameters),
      });

      setOpen(false);
      setParameters("{}");
      onRunCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to launch run");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      {trigger === undefined ? (
        <DialogTrigger asChild>
          <Button size="sm" className="gap-1">
            <Play className="h-3.5 w-3.5" />
            Run
          </Button>
        </DialogTrigger>
      ) : (
        trigger && <DialogTrigger asChild>{trigger}</DialogTrigger>
      )}
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Launch Run</DialogTitle>
          <DialogDescription>Execute workflow for experiment {experimentId}.</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="run-workflow" className="text-right">
                Workflow
              </Label>
              <Input
                id="run-workflow"
                value={workflowFile}
                disabled
                className="col-span-3 bg-muted"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="run-params" className="text-right">
                Parameters (JSON)
              </Label>
              <Textarea
                id="run-params"
                value={parameters}
                onChange={(e) => setParameters(e.target.value)}
                className="col-span-3 font-mono text-xs"
                rows={6}
              />
            </div>
            {error && <div className="text-sm text-red-500 col-span-4 text-center">{error}</div>}
          </div>
          <DialogFooter>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? "Launching..." : "Launch Run"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
