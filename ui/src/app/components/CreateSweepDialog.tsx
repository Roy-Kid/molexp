/**
 * Minimal sweep create: one param axis × values → N runs.
 * Opened from a parent menu / button (controlled dialog, no built-in trigger).
 */

import { useState } from "react";
import { workspaceApi } from "@/app/state/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "@/components/ui/toast";
import { WorkbenchAction } from "@/components/workbench";

interface CreateSweepDialogProps {
  projectId: string;
  experimentId: string;
  /** Called after runs are created; parent should refresh and open Runs tab. */
  onCreated: (count: number) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateSweepDialog({
  projectId,
  experimentId,
  onCreated,
  open,
  onOpenChange,
}: CreateSweepDialogProps): JSX.Element {
  const [paramKey, setParamKey] = useState("temperature");
  const [valuesText, setValuesText] = useState("300, 350, 400");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (): Promise<void> => {
    const key = paramKey.trim();
    if (!key) {
      setError("Name required");
      return;
    }
    const values = valuesText
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((raw) => {
        const n = Number(raw);
        return Number.isFinite(n) ? n : raw;
      });
    if (values.length < 2) {
      setError("Need ≥2 values");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      for (const value of values) {
        await workspaceApi.createRun(projectId, experimentId, {
          parameters: { [key]: value },
        });
      }
      onOpenChange(false);
      toast.success(`${values.length} runs`);
      onCreated(values.length);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Sweep</DialogTitle>
          <DialogDescription className="sr-only">One axis, one run per value.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3 py-2">
          <div className="grid gap-2">
            <Label htmlFor="sweep-key">Parameter</Label>
            <Input id="sweep-key" value={paramKey} onChange={(e) => setParamKey(e.target.value)} />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="sweep-values">Values</Label>
            <Input
              id="sweep-values"
              value={valuesText}
              onChange={(e) => setValuesText(e.target.value)}
              placeholder="300, 350, 400"
            />
          </div>
          {error && <p className="text-body-lg text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <WorkbenchAction
            kind="primary"
            size="default"
            disabled={busy}
            onClick={() => void submit()}
          >
            {busy ? "…" : "Create"}
          </WorkbenchAction>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
