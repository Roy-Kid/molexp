import { Plus } from "lucide-react";
import { useState } from "react";
import { workspaceApi } from "@/app/state/api";
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
import { WorkbenchAction, WorkbenchIconAction } from "@/components/workbench";

interface CreateProjectDialogProps {
  onProjectCreated: () => void;
  /** When set, the trigger is disabled and hover explains why. */
  writeDeniedReason?: string | null;
}

export function CreateProjectDialog({
  onProjectCreated,
  writeDeniedReason,
}: CreateProjectDialogProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (writeDeniedReason) return;
    setIsLoading(true);
    setError(null);

    try {
      await workspaceApi.createProject({
        name,
        description: "",
      });

      setOpen(false);
      setName("");
      onProjectCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create project");
    } finally {
      setIsLoading(false);
    }
  };

  if (writeDeniedReason) {
    return (
      <WorkbenchIconAction
        label="New project"
        kind="ghost"
        className="h-control-compact w-control-compact"
        aria-label="New project"
        deniedReason={writeDeniedReason}
      >
        <Plus className="h-4 w-4" />
      </WorkbenchIconAction>
    );
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <WorkbenchIconAction
          label="New project"
          kind="ghost"
          className="h-control-compact w-control-compact"
          aria-label="New project"
        >
          <Plus className="h-4 w-4" />
        </WorkbenchIconAction>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Create Project</DialogTitle>
          <DialogDescription>Create a new project to organize your experiments.</DialogDescription>
        </DialogHeader>
        <form onSubmit={(e) => void handleSubmit(e)}>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-4 sm:items-center sm:gap-4">
              <Label htmlFor="name" className="text-left sm:text-right">
                Name
              </Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="col-span-3"
                required
              />
            </div>
            {error && (
              <div className="text-body-lg text-status-failed-foreground col-span-4 text-center">
                {error}
              </div>
            )}
          </div>
          <DialogFooter>
            <WorkbenchAction kind="primary" size="default" type="submit" disabled={isLoading}>
              {isLoading ? "Creating..." : "Create Project"}
            </WorkbenchAction>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
