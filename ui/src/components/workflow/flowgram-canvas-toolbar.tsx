/**
 * FlowgramCanvasToolbar — save / discard chrome for the editable workflow
 * canvas. Pure shadcn/ui (Button, AlertDialog) — no FlowGram form-materials /
 * Semi Design. Shows an "Unsaved changes" cue while the canvas is dirty and
 * guards Discard behind a confirmation so edits aren't dropped by a mis-click.
 */

import { RotateCcw, Save } from "lucide-react";
import type { JSX } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button, buttonVariants } from "@/components/ui/button";

export interface FlowgramCanvasToolbarProps {
  onSave: () => void;
  /** Revert the canvas to the last saved version. */
  onDiscard: () => void;
  saving?: boolean;
  /** True once the canvas has unsaved edits. */
  dirty?: boolean;
}

export const FlowgramCanvasToolbar = ({
  onSave,
  onDiscard,
  saving = false,
  dirty = false,
}: FlowgramCanvasToolbarProps): JSX.Element => (
  <div className="flex items-center gap-2">
    {dirty && (
      <span
        role="status"
        className="flex items-center gap-1.5 text-xs font-medium text-warning-foreground"
      >
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-warning" />
        Unsaved changes
      </span>
    )}

    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={saving || !dirty}
          aria-label="Discard changes"
        >
          <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
          Discard
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Discard unsaved changes?</AlertDialogTitle>
          <AlertDialogDescription>
            This reverts the canvas to the last saved version. Your unsaved edits will be lost.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Keep editing</AlertDialogCancel>
          <AlertDialogAction
            className={buttonVariants({ variant: "destructive" })}
            onClick={onDiscard}
          >
            Discard changes
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>

    <Button
      type="button"
      size="sm"
      onClick={onSave}
      disabled={saving || !dirty}
      aria-label="Save workflow"
    >
      <Save className="mr-1.5 h-3.5 w-3.5" />
      {saving ? "Saving…" : "Save"}
    </Button>
  </div>
);
