import { Plus } from "lucide-react";
import type { ReactNode } from "react";
import { useCallback, useEffect, useState } from "react";
import type { TargetResponse } from "@/api/generated/models/TargetResponse";
import { TargetsService } from "@/api/generated/services/TargetsService";
import { AddTargetDialog } from "@/app/settings/AddTargetDialog";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

const NO_TARGET_VALUE = "__none__";

interface CreateExperimentDialogProps {
  projectId: string;
  onExperimentCreated: () => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  trigger?: ReactNode;
}

export function CreateExperimentDialog({
  projectId,
  onExperimentCreated,
  open: controlledOpen,
  onOpenChange,
  trigger,
}: CreateExperimentDialogProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const [name, setName] = useState("");
  const [workflow, setWorkflow] = useState("");
  const [description, setDescription] = useState("");
  const [parameterSpace, setParameterSpace] = useState("{}");
  const [defaultTarget, setDefaultTarget] = useState<string>(NO_TARGET_VALUE);
  const [targets, setTargets] = useState<TargetResponse[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = onOpenChange ?? setUncontrolledOpen;

  const refreshTargets = useCallback(async () => {
    try {
      const res = await TargetsService.listTargetsEndpointApiTargetsGet();
      setTargets(res.targets);
    } catch {
      setTargets([]);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    void refreshTargets();
  }, [open, refreshTargets]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      await workspaceApi.createExperiment(projectId, {
        name,
        workflow_source: workflow.trim() ? workflow : undefined,
        description,
        parameter_space: JSON.parse(parameterSpace),
        defaultTarget: defaultTarget === NO_TARGET_VALUE ? null : defaultTarget,
      });

      setOpen(false);
      setName("");
      setWorkflow("");
      setDescription("");
      setParameterSpace("{}");
      setDefaultTarget(NO_TARGET_VALUE);
      onExperimentCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create experiment");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      {trigger === undefined ? (
        <DialogTrigger asChild>
          <Button variant="outline" size="sm" className="h-7 gap-1">
            <Plus className="h-3.5 w-3.5" />
            <span className="sr-only sm:not-sr-only sm:whitespace-nowrap">New Experiment</span>
          </Button>
        </DialogTrigger>
      ) : (
        trigger && <DialogTrigger asChild>{trigger}</DialogTrigger>
      )}
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
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label htmlFor="exp-workflow" className="text-right">
                      Workflow
                    </Label>
                  </TooltipTrigger>
                  <TooltipContent side="left">
                    Optional. Leave blank to start with an empty canvas.
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <Input
                id="exp-workflow"
                value={workflow}
                onChange={(e) => setWorkflow(e.target.value)}
                placeholder="path/to/workflow.yaml"
                className="col-span-3"
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
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="exp-target" className="pt-2 text-right">
                Default target
              </Label>
              <div className="col-span-3 space-y-1.5">
                <Select value={defaultTarget} onValueChange={setDefaultTarget}>
                  <SelectTrigger id="exp-target">
                    <SelectValue placeholder="No default — pick at run time" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={NO_TARGET_VALUE}>No default</SelectItem>
                    {targets.map((t) => (
                      <SelectItem key={t.name} value={t.name}>
                        <span className="flex items-center gap-2">
                          <span className="font-medium">{t.name}</span>
                          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            {t.isRemote ? "remote" : "local"}
                          </span>
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <AddTargetDialog
                  trigger={
                    <button
                      type="button"
                      className="text-xs text-muted-foreground transition-colors hover:text-foreground"
                    >
                      + Add new target…
                    </button>
                  }
                  onCreated={(t) => {
                    void refreshTargets();
                    setDefaultTarget(t.name);
                  }}
                />
              </div>
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
