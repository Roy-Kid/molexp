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
  <div className="flex items-center gap-1">
    {dirty && (
      <span
        role="status"
        aria-label="Unsaved changes"
        title="Unsaved changes"
        className="mr-0.5 flex items-center"
      >
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-warning" />
      </span>
    )}

    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button
          type="button"
          size="icon"
          variant="ghost"
          disabled={saving || !dirty}
          aria-label="Discard changes"
          title="Discard changes"
          className="h-7 w-7"
        >
          <RotateCcw className="h-4 w-4" />
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
      size="icon"
      onClick={onSave}
      disabled={saving || !dirty}
      aria-label="Save workflow"
      title="Save workflow"
      className="h-7 w-7"
    >
      <Save className="h-4 w-4" />
    </Button>
  </div>
);
