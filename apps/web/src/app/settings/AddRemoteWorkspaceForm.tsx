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
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { WorkbenchAction } from "@/components/workbench";

const DEFAULT_CACHE_TTL_SECONDS = 300;

const emptyForm = (): WorkspaceTargetCreateRequest => ({
  name: "",
  host: "",
  root_path: "",
  port: null,
  identity_file: null,
  ssh_opts: [],
  cache_dir: null,
  cache_ttl_seconds: DEFAULT_CACHE_TTL_SECONDS,
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
        cache_dir: form.cache_dir?.trim() ? form.cache_dir.trim() : null,
        cache_ttl_seconds:
          form.cache_ttl_seconds == null
            ? DEFAULT_CACHE_TTL_SECONDS
            : Number(form.cache_ttl_seconds),
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
    variant === "card" ? "self-start space-y-3 border-t border-border/60 pt-4" : "space-y-3";

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
        <Collapsible>
          <CollapsibleTrigger className="text-label text-muted-foreground">
            Cache (lazy-download mirror)
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-3 pt-3">
            <div className="space-y-1">
              <Label htmlFor="add-remote-ws-cache-ttl">Cache TTL (seconds)</Label>
              <Input
                id="add-remote-ws-cache-ttl"
                type="number"
                min={0}
                value={form.cache_ttl_seconds ?? DEFAULT_CACHE_TTL_SECONDS}
                onChange={(e) =>
                  setForm({
                    ...form,
                    cache_ttl_seconds:
                      e.target.value === "" ? DEFAULT_CACHE_TTL_SECONDS : Number(e.target.value),
                  })
                }
                placeholder={String(DEFAULT_CACHE_TTL_SECONDS)}
              />
              <p className="text-label text-muted-foreground">
                How long a cached file/dir entry stays fresh. 0 always re-stats the remote FS but
                still serves mirror bytes when mtime matches.
              </p>
            </div>
            <div className="space-y-1">
              <Label htmlFor="add-remote-ws-cache-dir">Cache directory (optional)</Label>
              <Input
                id="add-remote-ws-cache-dir"
                value={form.cache_dir ?? ""}
                onChange={(e) =>
                  setForm({ ...form, cache_dir: e.target.value === "" ? null : e.target.value })
                }
                placeholder="~/.molexp/remote_cache/<name>"
              />
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
      {error && <p className="text-body-lg text-status-failed-foreground">{error}</p>}
      <div className="flex justify-end gap-2">
        {onCancel && (
          <WorkbenchAction kind="ghost" size="default" type="button" onClick={onCancel}>
            Cancel
          </WorkbenchAction>
        )}
        <WorkbenchAction kind="primary" size="default" type="submit" disabled={submitting}>
          {submitting ? "Adding…" : "Add remote workspace"}
        </WorkbenchAction>
      </div>
    </form>
  );
}
