/**
 * Modal dialog wrapping AddTargetForm — used for inline "+ Add target..."
 * shortcuts inside Create Experiment / Create Run flows so users don't have
 * to leave their current task to register a new compute target.
 */

import type { ReactNode } from "react";
import { useState } from "react";
import type { TargetResponse } from "@/api/generated/models/TargetResponse";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import { AddTargetForm } from "./AddTargetForm";

interface AddTargetDialogProps {
  trigger: ReactNode;
  onCreated?: (target: TargetResponse) => void;
}

export function AddTargetDialog({ trigger, onCreated }: AddTargetDialogProps): JSX.Element {
  const [open, setOpen] = useState(false);

  const handleCreated = (target: TargetResponse): void => {
    setOpen(false);
    onCreated?.(target);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-[480px]">
        <DialogHeader>
          <DialogTitle>Add compute target</DialogTitle>
          <DialogDescription>
            Register a destination for runs. Local targets execute on this machine; remote targets
            dispatch via SSH and an optional batch scheduler.
          </DialogDescription>
        </DialogHeader>
        <AddTargetForm variant="plain" onCreated={handleCreated} onCancel={() => setOpen(false)} />
      </DialogContent>
    </Dialog>
  );
}
