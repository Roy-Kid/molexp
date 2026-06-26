import { Plus } from "lucide-react";
import type { ReactNode } from "react";
import { useCallback, useEffect, useState } from "react";
import type { TargetResponse } from "@/api/generated/models/TargetResponse";
import { ExperimentsService } from "@/api/generated/services/ExperimentsService";
import { TargetsService } from "@/api/generated/services/TargetsService";
import { ParametersForm } from "@/app/runs/ParametersForm";
import {
  type InputField,
  parseInputSchema,
  SchemaForm,
  schemaDefaults,
} from "@/app/runs/SchemaForm";
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

const NO_TARGET_VALUE = "__none__";

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
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  const [inputSchema, setInputSchema] = useState<InputField[] | null>(null);
  const [target, setTarget] = useState<string>(NO_TARGET_VALUE);
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
    void ExperimentsService.getExperimentApiProjectsProjectIdExperimentsExperimentIdGet(
      projectId,
      experimentId,
    )
      .then((exp) => {
        if (exp.defaultTarget) setTarget(exp.defaultTarget);
        const schema = parseInputSchema(exp.workflow);
        setInputSchema(schema);
        if (schema) setParameters(schemaDefaults(schema));
      })
      .catch(() => {
        // experiment may not yet be readable; ignore
      });
  }, [open, projectId, experimentId, refreshTargets]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      await workspaceApi.createRun(projectId, experimentId, {
        parameters,
        target: target === NO_TARGET_VALUE ? null : target,
      });

      setOpen(false);
      setParameters({});
      setTarget(NO_TARGET_VALUE);
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
            <Plus className="h-3.5 w-3.5" />
            New run
          </Button>
        </DialogTrigger>
      ) : (
        trigger && <DialogTrigger asChild>{trigger}</DialogTrigger>
      )}
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>New run</DialogTitle>
          <DialogDescription>
            Create a run for experiment {experimentId}. With a compute target it starts immediately;
            without one it is created <span className="font-medium">pending</span> and started via{" "}
            <code className="font-mono">molexp run</code> (or the run's Start action).
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-4 sm:items-center sm:gap-4">
              <Label htmlFor="run-workflow" className="text-left sm:text-right">
                Workflow
              </Label>
              <Input
                id="run-workflow"
                value={workflowFile}
                disabled
                className="col-span-3 bg-muted"
              />
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-4 sm:items-start sm:gap-4">
              <Label className="pt-2 text-left sm:text-right">
                {inputSchema ? "Inputs" : "Parameters"}
              </Label>
              <div className="col-span-3">
                {inputSchema ? (
                  <SchemaForm schema={inputSchema} value={parameters} onChange={setParameters} />
                ) : (
                  <ParametersForm value={parameters} onChange={setParameters} />
                )}
              </div>
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-4 sm:items-start sm:gap-4">
              <Label htmlFor="run-target" className="pt-2 text-left sm:text-right">
                Target
              </Label>
              <div className="col-span-3 space-y-1.5">
                <Select value={target} onValueChange={setTarget}>
                  <SelectTrigger id="run-target">
                    <SelectValue placeholder="No target — local in-process" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={NO_TARGET_VALUE}>No target (local)</SelectItem>
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
                    setTarget(t.name);
                  }}
                />
              </div>
            </div>
            {error && <div className="text-sm text-red-500 col-span-4 text-center">{error}</div>}
          </div>
          <DialogFooter>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? "Creating..." : "Create run"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
