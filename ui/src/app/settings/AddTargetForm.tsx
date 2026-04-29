/**
 * Reusable form for creating a ComputeTarget.  Used both inside the Settings
 * page (Compute targets section) and inline as a modal sub-dialog from the
 * Create Experiment / Create Run flows.
 */

import { useState } from "react";
import { TargetCreateRequest } from "@/api/generated/models/TargetCreateRequest";
import type { TargetResponse } from "@/api/generated/models/TargetResponse";
import { TargetsService } from "@/api/generated/services/TargetsService";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type Scheduler = TargetCreateRequest.scheduler;

const SCHEDULERS: Scheduler[] = [
  TargetCreateRequest.scheduler.SHELL,
  TargetCreateRequest.scheduler.SLURM,
  TargetCreateRequest.scheduler.PBS,
  TargetCreateRequest.scheduler.LSF,
];

const schedulerLabel: Record<Scheduler, string> = {
  [TargetCreateRequest.scheduler.SHELL]: "Local shell",
  [TargetCreateRequest.scheduler.SLURM]: "SLURM",
  [TargetCreateRequest.scheduler.PBS]: "PBS",
  [TargetCreateRequest.scheduler.LSF]: "LSF",
};

const emptyForm = (): TargetCreateRequest => ({
  name: "",
  scratchRoot: "",
  scheduler: TargetCreateRequest.scheduler.SHELL,
  host: null,
  port: null,
  identityFile: null,
  sshOpts: [],
});

interface AddTargetFormProps {
  onCreated?: (target: TargetResponse) => void;
  onCancel?: () => void;
  variant?: "card" | "plain";
}

export function AddTargetForm({
  onCreated,
  onCancel,
  variant = "card",
}: AddTargetFormProps): JSX.Element {
  const [form, setForm] = useState<TargetCreateRequest>(emptyForm());
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const payload: TargetCreateRequest = {
        ...form,
        host: form.host?.trim() ? form.host.trim() : null,
        port: form.port == null ? null : Number(form.port),
        identityFile: form.identityFile?.trim() ? form.identityFile.trim() : null,
      };
      const created = await TargetsService.createTargetEndpointApiTargetsPost(payload);
      setForm(emptyForm());
      onCreated?.(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create target");
    } finally {
      setSubmitting(false);
    }
  };

  const formClass =
    variant === "card"
      ? "space-y-3 rounded-md border border-border bg-muted/20 p-4 self-start"
      : "space-y-3";

  return (
    <form onSubmit={handleSubmit} className={formClass}>
      {variant === "card" && <h3 className="text-sm font-semibold text-foreground">Add target</h3>}
      <div className="space-y-3">
        <div className="space-y-1">
          <Label htmlFor="add-target-name">Name</Label>
          <Input
            id="add-target-name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="e.g. hpc-slurm"
            required
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="add-target-scheduler">Scheduler</Label>
          <Select
            value={form.scheduler ?? TargetCreateRequest.scheduler.SHELL}
            onValueChange={(v) => setForm({ ...form, scheduler: v as Scheduler })}
          >
            <SelectTrigger id="add-target-scheduler">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SCHEDULERS.map((s) => (
                <SelectItem key={s} value={s}>
                  {schedulerLabel[s]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="add-target-scratch">Scratch root</Label>
          <Input
            id="add-target-scratch"
            value={form.scratchRoot}
            onChange={(e) => setForm({ ...form, scratchRoot: e.target.value })}
            placeholder="/scratch/me/molexp"
            required
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="add-target-host">
            Host <span className="text-muted-foreground">(blank = local)</span>
          </Label>
          <Input
            id="add-target-host"
            value={form.host ?? ""}
            onChange={(e) => setForm({ ...form, host: e.target.value })}
            placeholder="me@hpc.example.org"
          />
        </div>
        {form.host?.trim() ? (
          <>
            <div className="space-y-1">
              <Label htmlFor="add-target-port">SSH port</Label>
              <Input
                id="add-target-port"
                type="number"
                value={form.port ?? ""}
                onChange={(e) =>
                  setForm({
                    ...form,
                    port: e.target.value === "" ? null : Number(e.target.value),
                  })
                }
                placeholder="22"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="add-target-identity">Identity file</Label>
              <Input
                id="add-target-identity"
                value={form.identityFile ?? ""}
                onChange={(e) => setForm({ ...form, identityFile: e.target.value })}
                placeholder="~/.ssh/id_ed25519"
              />
            </div>
          </>
        ) : null}
      </div>
      {error && <p className="text-sm text-red-500">{error}</p>}
      <div className="flex justify-end gap-2">
        {onCancel && (
          <Button type="button" variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button type="submit" disabled={submitting}>
          {submitting ? "Adding…" : "Add target"}
        </Button>
      </div>
    </form>
  );
}
