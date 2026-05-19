/**
 * Settings tab for managing remote workspace descriptors — the entries
 * that point the server's active workspace at a remote SSH root.
 * Mirrors ComputeTargetsPanel's visual structure (header + list + inline
 * Test result + sidebar Add form) but operates on the workspace-target
 * registry (POST /api/workspace/targets) and POST /api/workspace/open
 * with `kind: "remote"`.
 */

import { Check, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import type { TargetTestResponse } from "@/api/generated/models/TargetTestResponse";
import type { WorkspaceTargetResponse } from "@/api/generated/models/WorkspaceTargetResponse";
import { WorkspaceService } from "@/api/generated/services/WorkspaceService";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

import { emitWorkspaceSwitching } from "../state/workspaceSwitchEvents";

import { AddRemoteWorkspaceDialog } from "./AddRemoteWorkspaceDialog";

export function RemoteWorkspacesPanel(): JSX.Element {
  const [targets, setTargets] = useState<WorkspaceTargetResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  const [busy, setBusy] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TargetTestResponse | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [activeName, setActiveName] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setListError(null);
    try {
      const res = await WorkspaceService.listWorkspaceTargetsApiWorkspaceTargetsGet();
      setTargets(res.targets);
    } catch (err) {
      setListError(err instanceof Error ? err.message : "Failed to list remote workspaces");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const handleDelete = async (name: string): Promise<void> => {
    setBusy(name);
    setActionError(null);
    setTestResult(null);
    try {
      await WorkspaceService.deleteWorkspaceTargetApiWorkspaceTargetsNameDelete(name);
      if (activeName === name) {
        setActiveName(null);
      }
      await refresh();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to delete remote workspace");
    } finally {
      setBusy(null);
    }
  };

  const handleTest = async (name: string): Promise<void> => {
    setBusy(name);
    setActionError(null);
    setTestResult(null);
    try {
      const res = await WorkspaceService.testWorkspaceTargetApiWorkspaceTargetsNameTestPost(name);
      setTestResult(res);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to test remote workspace");
    } finally {
      setBusy(null);
    }
  };

  const handleSetActive = async (name: string): Promise<void> => {
    setBusy(name);
    setActionError(null);
    setTestResult(null);
    try {
      await WorkspaceService.openWorkspaceApiWorkspaceOpenPost({
        kind: "remote",
        name,
      });
      emitWorkspaceSwitching({ activeDescriptor: name });
      setActiveName(name);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to switch active workspace");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-3">
      <header className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-foreground">
            Remote workspaces <span className="text-muted-foreground">({targets.length})</span>
          </h3>
          <p className="text-xs text-muted-foreground">
            SSH-reachable workspace roots. Set one as Active to mount it as the current workspace.
          </p>
        </div>
        <AddRemoteWorkspaceDialog
          trigger={
            <Button size="sm" variant="default">
              + Add remote workspace
            </Button>
          }
          onCreated={() => void refresh()}
        />
      </header>
      {listError && <p className="text-sm text-red-500">{listError}</p>}
      {loading && targets.length === 0 ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : targets.length === 0 ? (
        <p className="rounded-md border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
          No remote workspaces registered. Add one to mount a workspace hosted on an HPC node.
        </p>
      ) : (
        <ul className="divide-y divide-border rounded-md border border-border">
          {targets.map((t) => {
            const isActive = t.name === activeName;
            return (
              <li key={t.name} className="flex items-center gap-3 px-3 py-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-medium truncate">{t.name}</span>
                    <Badge variant={isActive ? "default" : "outline"}>
                      {isActive ? "Active" : "Inactive"}
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {t.host} → {t.root_path}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busy === t.name}
                  onClick={() => void handleTest(t.name)}
                >
                  Test
                </Button>
                <Button
                  variant={isActive ? "secondary" : "outline"}
                  size="sm"
                  disabled={busy === t.name || isActive}
                  onClick={() => void handleSetActive(t.name)}
                >
                  {isActive ? "Active" : "Set active"}
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove ${t.name}`}
                  title={
                    isActive ? "Switch to another workspace first" : `Remove ${t.name}`
                  }
                  disabled={busy === t.name || isActive}
                  onClick={() => void handleDelete(t.name)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </li>
            );
          })}
        </ul>
      )}
      {actionError && <p className="text-sm text-red-500">{actionError}</p>}
      {testResult && (
        <div className="rounded-md border border-border bg-muted/30 p-3 text-sm space-y-1">
          <div className="flex items-center gap-2 font-medium">
            {testResult.ok ? (
              <Check className="h-4 w-4 text-green-500" />
            ) : (
              <X className="h-4 w-4 text-red-500" />
            )}
            <span>{testResult.name}</span>
            <span className="text-muted-foreground">
              {testResult.ok ? "reachable" : "unreachable"}
            </span>
          </div>
          {testResult.error && <p className="text-xs text-red-500">{testResult.error}</p>}
          <ul className="space-y-0.5 pl-1">
            {testResult.checks.map((c) => (
              <li
                key={c.label}
                className="flex items-start gap-1.5 text-xs text-muted-foreground"
              >
                {c.ok ? (
                  <Check className="h-3 w-3 mt-0.5 text-green-500 flex-shrink-0" />
                ) : (
                  <X className="h-3 w-3 mt-0.5 text-red-500 flex-shrink-0" />
                )}
                <span>
                  {c.label}
                  {c.detail && <span className="text-red-500"> — {c.detail}</span>}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
