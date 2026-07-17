/**
 * Minimal sweep create: one param axis × values → N runs.
 */

import { Grid3x3 } from "lucide-react";
import { useState } from "react";
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
import { toast } from "@/components/ui/toast";

interface CreateSweepDialogProps {
  projectId: string;
  experimentId: string;
  /** Called after runs are created; parent should refresh and open Runs tab. */
  onCreated: (count: number) => void;
}

export function CreateSweepDialog({
  projectId,
  experimentId,
  onCreated,
}: CreateSweepDialogProps): JSX.Element {
  const [open, setOpen] = useState(false);
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
      setOpen(false);
      toast.success(`${values.length} runs`);
      onCreated(values.length);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button size="sm" variant="outline" className="h-7 gap-1">
          <Grid3x3 className="h-3.5 w-3.5" />
          Sweep
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle>Sweep</DialogTitle>
          <DialogDescription className="sr-only">One axis, one run per value.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3 py-2">
          <div className="grid gap-1.5">
            <Label htmlFor="sweep-key">Parameter</Label>
            <Input id="sweep-key" value={paramKey} onChange={(e) => setParamKey(e.target.value)} />
          </div>
          <div className="grid gap-1.5">
            <Label htmlFor="sweep-values">Values</Label>
            <Input
              id="sweep-values"
              value={valuesText}
              onChange={(e) => setValuesText(e.target.value)}
              placeholder="300, 350, 400"
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button disabled={busy} onClick={() => void submit()}>
            {busy ? "…" : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
