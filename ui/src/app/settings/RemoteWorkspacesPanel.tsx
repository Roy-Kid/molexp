/**
 * Settings tab for managing remote workspace descriptors — the entries
 * that point the server's active workspace at a remote SSH root.
 * Mirrors ComputeTargetsPanel's visual structure (header + list + inline
 * Test result + sidebar Add form) but operates on the workspace-target
 * registry (POST /api/workspace/targets) and POST /api/workspace/open
 * with `kind: "remote"`.
 */

import { AlertTriangle, Check, RefreshCw, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import type { TargetTestResponse } from "@/api/generated/models/TargetTestResponse";
import type { WorkspaceTargetResponse } from "@/api/generated/models/WorkspaceTargetResponse";
import { WorkspaceService } from "@/api/generated/services/WorkspaceService";
import { WorkbenchAction, WorkbenchIconAction, WorkbenchTag } from "@/components/workbench";
import { emitWorkspaceSwitching } from "../state/workspaceSwitchEvents";
import { AddRemoteWorkspaceDialog } from "./AddRemoteWorkspaceDialog";

interface CacheStatus {
  dropped: number;
  fetchedAt: number;
}

export function RemoteWorkspacesPanel(): JSX.Element {
  const [targets, setTargets] = useState<WorkspaceTargetResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  const [busy, setBusy] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TargetTestResponse | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [activeName, setActiveName] = useState<string | null>(null);
  const [openWarnings, setOpenWarnings] = useState<string[]>([]);
  const [cacheStatus, setCacheStatus] = useState<CacheStatus | null>(null);

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
    setOpenWarnings([]);
    setCacheStatus(null);
    try {
      const info = await WorkspaceService.openWorkspaceApiWorkspaceOpenPost({
        kind: "remote",
        name,
      });
      emitWorkspaceSwitching({ activeDescriptor: name });
      setActiveName(name);
      setOpenWarnings(info.warnings ?? []);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Failed to switch active workspace");
    } finally {
      setBusy(null);
    }
  };

  const handleRefreshCache = async (name: string): Promise<void> => {
    setBusy(name);
    setActionError(null);
    setTestResult(null);
    try {
      const res = await WorkspaceService.refreshWorkspaceCacheApiWorkspaceCacheRefreshPost({
        scope: "indices",
      });
      setCacheStatus({ dropped: res.dropped, fetchedAt: Date.now() });
      setOpenWarnings(res.warnings ?? []);
    } catch (err) {
      setActionError(
        err instanceof Error ? err.message : "Failed to refresh remote workspace cache",
      );
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-3">
      <header className="flex items-center justify-between gap-3">
        <div>
          <h3 className="sr-only">Remote workspaces</h3>
          <p className="text-xs text-muted-foreground">
            {targets.length} registered · Set an SSH-reachable root as Active to mount it as the
            current workspace.
          </p>
        </div>
        <AddRemoteWorkspaceDialog
          trigger={
            <WorkbenchAction kind="primary" size="compact">
              + Add remote workspace
            </WorkbenchAction>
          }
          onCreated={() => void refresh()}
        />
      </header>
      {listError && <p className="text-sm text-status-failed-foreground">{listError}</p>}
      {loading && targets.length === 0 ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : targets.length === 0 ? (
        <p className="border-y border-dashed border-border/70 py-6 text-center text-sm text-muted-foreground">
          No remote workspaces registered. Add one to mount a workspace hosted on an HPC node.
        </p>
      ) : (
        <ul className="divide-y divide-border border-y border-border">
          {targets.map((t) => {
            const isActive = t.name === activeName;
            return (
              <li key={t.name} className="flex items-center gap-3 px-3 py-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-medium truncate">{t.name}</span>
                    <WorkbenchTag meaning={isActive ? "selection" : "metadata"}>
                      {isActive ? "Active" : "Inactive"}
                    </WorkbenchTag>
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {t.host} → {t.root_path}
                  </div>
                </div>
                <WorkbenchAction
                  kind="ghost"
                  size="compact"
                  disabled={busy === t.name}
                  onClick={() => void handleTest(t.name)}
                >
                  Test
                </WorkbenchAction>
                <WorkbenchAction
                  kind="primary"
                  size="compact"
                  disabled={busy === t.name || isActive}
                  onClick={() => void handleSetActive(t.name)}
                >
                  {isActive ? "Active" : "Set active"}
                </WorkbenchAction>
                {isActive && (
                  <WorkbenchAction
                    kind="ghost"
                    size="compact"
                    aria-label={`Refresh ${t.name}`}
                    title={`Refresh navigation cache (TTL ${t.cache_ttl_seconds ?? 300}s)`}
                    disabled={busy === t.name}
                    onClick={() => void handleRefreshCache(t.name)}
                  >
                    <RefreshCw className="h-4 w-4" />
                    <span className="ml-2">Refresh</span>
                  </WorkbenchAction>
                )}
                <WorkbenchIconAction
                  label={`Remove ${t.name}`}
                  kind="ghost"
                  title={isActive ? "Switch to another workspace first" : `Remove ${t.name}`}
                  disabled={busy === t.name || isActive}
                  onClick={() => void handleDelete(t.name)}
                >
                  <Trash2 className="h-4 w-4" />
                </WorkbenchIconAction>
              </li>
            );
          })}
        </ul>
      )}
      {actionError && <p className="text-sm text-status-failed-foreground">{actionError}</p>}
      {cacheStatus && (
        <p className="text-xs text-muted-foreground">
          Refreshed navigation cache — dropped {cacheStatus.dropped}{" "}
          {cacheStatus.dropped === 1 ? "entry" : "entries"}.
        </p>
      )}
      {openWarnings.length > 0 && (
        <div className="border-y border-status-warning/40 bg-status-warning-soft px-3 py-3 text-sm">
          <div className="mb-1 flex items-center gap-2 text-status-warning-foreground">
            <AlertTriangle className="h-4 w-4" />
            <span className="font-medium">
              {openWarnings.length} {openWarnings.length === 1 ? "warning" : "warnings"} while
              fetching the navigation tree
            </span>
          </div>
          <ul className="space-y-1 pl-1 text-xs text-muted-foreground">
            {openWarnings.map((w) => (
              <li key={w} className="break-all">
                {w}
              </li>
            ))}
          </ul>
        </div>
      )}
      {testResult && (
        <div className="space-y-1 border-y border-border/60 bg-muted/30 px-3 py-3 text-sm">
          <div className="flex items-center gap-2 font-medium">
            {testResult.ok ? (
              <Check className="h-4 w-4 text-status-completed-foreground" />
            ) : (
              <X className="h-4 w-4 text-status-failed-foreground" />
            )}
            <span>{testResult.name}</span>
            <span className="text-muted-foreground">
              {testResult.ok ? "reachable" : "unreachable"}
            </span>
          </div>
          {testResult.error && (
            <p className="text-xs text-status-failed-foreground">{testResult.error}</p>
          )}
          <ul className="space-y-1 pl-1">
            {testResult.checks.map((c) => (
              <li key={c.label} className="flex items-start gap-2 text-xs text-muted-foreground">
                {c.ok ? (
                  <Check className="h-3 w-3 mt-1 text-status-completed-foreground flex-shrink-0" />
                ) : (
                  <X className="h-3 w-3 mt-1 text-status-failed-foreground flex-shrink-0" />
                )}
                <span>
                  {c.label}
                  {c.detail && <span className="text-status-failed-foreground"> — {c.detail}</span>}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
