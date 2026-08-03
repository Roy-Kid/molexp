/**
 * FlowgramCanvasToolbar — save / discard chrome for the editable workflow canvas.
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
import { WorkbenchIconAction } from "@/components/workbench";

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
        className="mr-1 flex items-center"
      >
        <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-status-warning" />
      </span>
    )}

    <AlertDialog>
      <AlertDialogTrigger asChild>
        <WorkbenchIconAction label="Discard changes" disabled={saving || !dirty}>
          <RotateCcw className="h-4 w-4" />
        </WorkbenchIconAction>
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
          <AlertDialogAction intent="danger" onClick={onDiscard}>
            Discard changes
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>

    <WorkbenchIconAction
      label="Save workflow"
      kind="secondary"
      disabled={saving || !dirty}
      onClick={onSave}
    >
      <Save className="h-4 w-4" />
    </WorkbenchIconAction>
  </div>
);
