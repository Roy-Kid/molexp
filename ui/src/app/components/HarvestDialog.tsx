/**
 * Harvest a terminal run into Knowledge — Dialog form (replaces window.prompt).
 */

import { BookMarked } from "lucide-react";
import { type JSX, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/components/ui/toast";
import { WorkbenchAction, WorkbenchIconAction } from "@/components/workbench";

const KINDS = ["Finding", "FailureAnalysis", "Note", "Hypothesis"] as const;

interface HarvestDialogProps {
  projectId: string;
  experimentId: string;
  runId: string;
  onHarvested: (path: string) => void;
  trigger?: JSX.Element;
}

export function HarvestDialog({
  projectId,
  experimentId,
  runId,
  onHarvested,
  trigger,
}: HarvestDialogProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const [kind, setKind] = useState<string>("Finding");
  const [narrative, setNarrative] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (): Promise<void> => {
    const text = narrative.trim();
    if (!text) {
      setError("Narrative required.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(experimentId)}/runs/${encodeURIComponent(runId)}/harvest`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            kind,
            narrative: text,
            created_by: "ui",
          }),
        },
      );
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || res.statusText);
      }
      const body = (await res.json()) as { name?: string; path?: string };
      const path = body.path?.trim() || "";
      setOpen(false);
      setNarrative("");
      toast.success("Harvested");
      onHarvested(path);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) {
          setError(null);
        }
      }}
    >
      <DialogTrigger asChild>
        {trigger ?? (
          <WorkbenchIconAction label="Harvest to knowledge">
            <BookMarked className="h-3.5 w-3.5" />
          </WorkbenchIconAction>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Harvest</DialogTitle>
        </DialogHeader>
        <div className="grid gap-3 py-1">
          <div className="grid gap-2">
            <Label htmlFor="harvest-kind">Kind</Label>
            <Select value={kind} onValueChange={setKind}>
              <SelectTrigger id="harvest-kind">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {KINDS.map((k) => (
                  <SelectItem key={k} value={k}>
                    {k}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-2">
            <Label htmlFor="harvest-narrative">Narrative</Label>
            <Textarea
              id="harvest-narrative"
              value={narrative}
              onChange={(e) => setNarrative(e.target.value)}
              rows={4}
              placeholder="What does this run mean?"
              disabled={busy}
            />
          </div>
          {error && <p className="text-body-lg text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <WorkbenchAction
            kind="ghost"
            size="compact"
            disabled={busy}
            onClick={() => setOpen(false)}
          >
            Cancel
          </WorkbenchAction>
          <WorkbenchAction
            kind="primary"
            size="compact"
            disabled={busy || !narrative.trim()}
            onClick={() => void submit()}
          >
            {busy ? "Saving…" : "Save"}
          </WorkbenchAction>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
