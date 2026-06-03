/**
 * FlowgramCanvasToolbar — minimal save chrome for the editable workflow canvas.
 * Pure shadcn/ui (Button) — no FlowGram form-materials / Semi Design.
 */

import { Save } from "lucide-react";
import type { JSX } from "react";
import { Button } from "@/components/ui/button";

export interface FlowgramCanvasToolbarProps {
  onSave: () => void;
  saving?: boolean;
  /** True once the canvas has unsaved edits. */
  dirty?: boolean;
}

export const FlowgramCanvasToolbar = ({
  onSave,
  saving = false,
  dirty = false,
}: FlowgramCanvasToolbarProps): JSX.Element => (
  <div className="flex items-center gap-2">
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
