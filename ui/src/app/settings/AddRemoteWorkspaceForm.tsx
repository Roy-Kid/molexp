/**
 * Form for registering a remote-workspace descriptor.  Mirrors the
 * AddTargetForm pattern — controlled-state input + service call —
 * but persists into the *workspace-target* registry rather than the
 * per-workspace ComputeTarget registry.
 */

import { useState } from "react";

import type { WorkspaceTargetCreateRequest } from "@/api/generated/models/WorkspaceTargetCreateRequest";
import type { WorkspaceTargetResponse } from "@/api/generated/models/WorkspaceTargetResponse";
import { WorkspaceService } from "@/api/generated/services/WorkspaceService";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const emptyForm = (): WorkspaceTargetCreateRequest => ({
  name: "",
  host: "",
  root_path: "",
  port: null,
  identity_file: null,
  ssh_opts: [],
});

interface AddRemoteWorkspaceFormProps {
  onCreated?: (target: WorkspaceTargetResponse) => void;
  onCancel?: () => void;
  variant?: "card" | "plain";
}

export function AddRemoteWorkspaceForm({
  onCreated,
  onCancel,
  variant = "plain",
}: AddRemoteWorkspaceFormProps): JSX.Element {
  const [form, setForm] = useState<WorkspaceTargetCreateRequest>(emptyForm());
  const [sshOptsRaw, setSshOptsRaw] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const payload: WorkspaceTargetCreateRequest = {
        ...form,
        name: form.name.trim(),
        host: form.host.trim(),
        root_path: form.root_path.trim(),
        port: form.port == null ? null : Number(form.port),
        identity_file: form.identity_file?.trim() ? form.identity_file.trim() : null,
        ssh_opts: sshOptsRaw
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
      };
      const created = await WorkspaceService.createWorkspaceTargetApiWorkspaceTargetsPost(payload);
      setForm(emptyForm());
      setSshOptsRaw("");
      onCreated?.(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to register remote workspace");
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
      <div className="space-y-3">
        <div className="space-y-1">
          <Label htmlFor="add-remote-ws-name">Name</Label>
          <Input
            id="add-remote-ws-name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="e.g. prod-cluster"
            required
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="add-remote-ws-host">Host</Label>
          <Input
            id="add-remote-ws-host"
            value={form.host}
            onChange={(e) => setForm({ ...form, host: e.target.value })}
            placeholder="me@hpc.example.org"
            required
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="add-remote-ws-root">Root path</Label>
          <Input
            id="add-remote-ws-root"
            value={form.root_path}
            onChange={(e) => setForm({ ...form, root_path: e.target.value })}
            placeholder="/scratch/me/molexp-lab"
            required
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="add-remote-ws-port">SSH port</Label>
          <Input
            id="add-remote-ws-port"
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
          <Label htmlFor="add-remote-ws-identity">Identity file</Label>
          <Input
            id="add-remote-ws-identity"
            value={form.identity_file ?? ""}
            onChange={(e) => setForm({ ...form, identity_file: e.target.value })}
            placeholder="~/.ssh/id_ed25519"
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="add-remote-ws-ssh-opts">SSH opts (comma-separated)</Label>
          <Input
            id="add-remote-ws-ssh-opts"
            value={sshOptsRaw}
            onChange={(e) => setSshOptsRaw(e.target.value)}
            placeholder="-o, StrictHostKeyChecking=accept-new"
          />
        </div>
      </div>
      {error && <p className="text-sm text-red-500">{error}</p>}
      <div className="flex justify-end gap-2">
        {onCancel && (
          <Button type="button" variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button type="submit" disabled={submitting}>
          {submitting ? "Adding…" : "Add remote workspace"}
        </Button>
      </div>
    </form>
  );
}
