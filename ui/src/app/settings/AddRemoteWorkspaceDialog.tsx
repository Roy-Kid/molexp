/**
 * Modal dialog wrapping AddRemoteWorkspaceForm.  Mirrors AddTargetDialog
 * for consistency with the Compute targets flow.
 */

import type { ReactNode } from "react";
import { useState } from "react";

import type { WorkspaceTargetResponse } from "@/api/generated/models/WorkspaceTargetResponse";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import { AddRemoteWorkspaceForm } from "./AddRemoteWorkspaceForm";

interface AddRemoteWorkspaceDialogProps {
  trigger: ReactNode;
  onCreated?: (target: WorkspaceTargetResponse) => void;
}

export function AddRemoteWorkspaceDialog({
  trigger,
  onCreated,
}: AddRemoteWorkspaceDialogProps): JSX.Element {
  const [open, setOpen] = useState(false);

  const handleCreated = (target: WorkspaceTargetResponse): void => {
    setOpen(false);
    onCreated?.(target);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>Add remote workspace</DialogTitle>
          <DialogDescription>
            Register a remote workspace root reachable over SSH. The remote directory must already
            exist — this descriptor only tells molexp how to reach it.
          </DialogDescription>
        </DialogHeader>
        <AddRemoteWorkspaceForm
          variant="plain"
          onCreated={handleCreated}
          onCancel={() => setOpen(false)}
        />
      </DialogContent>
    </Dialog>
  );
}
