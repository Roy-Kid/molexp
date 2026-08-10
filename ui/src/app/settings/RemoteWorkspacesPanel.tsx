/**
 * Settings section: remote workspace descriptors.
 * List/mutations via TanStack Query (no hand-rolled useEffect fetch).
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Check,
  FlaskConical,
  Plus,
  Power,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { useState } from "react";

import type { TargetTestResponse } from "@/api/generated/models/TargetTestResponse";
import { WorkspaceService } from "@/api/generated/services/WorkspaceService";
import { usePermissions } from "@/app/auth";
import { WorkbenchIconAction, WorkbenchTag } from "@/components/workbench";
import { emitWorkspaceSwitching } from "../state/workspaceSwitchEvents";
import { AddRemoteWorkspaceDialog } from "./AddRemoteWorkspaceDialog";
import { settingsKeys } from "./settingsKeys";

interface CacheStatus {
  dropped: number;
  fetchedAt: number;
}

export function RemoteWorkspacesPanel(): JSX.Element {
  const { writeDeniedReason } = usePermissions();
  const queryClient = useQueryClient();
  const [busy, setBusy] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<TargetTestResponse | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [activeName, setActiveName] = useState<string | null>(null);
  const [openWarnings, setOpenWarnings] = useState<string[]>([]);
  const [cacheStatus, setCacheStatus] = useState<CacheStatus | null>(null);

  const listQuery = useQuery({
    queryKey: settingsKeys.remoteWorkspaces(),
    queryFn: async () => {
      const res = await WorkspaceService.listWorkspaceTargetsApiWorkspaceTargetsGet();
      return res.targets;
    },
  });

  const targets = listQuery.data ?? [];
  const invalidate = async (): Promise<void> => {
    await queryClient.invalidateQueries({ queryKey: settingsKeys.remoteWorkspaces() });
  };

  const deleteMutation = useMutation({
    mutationFn: (name: string) =>
      WorkspaceService.deleteWorkspaceTargetApiWorkspaceTargetsNameDelete(name),
    onSuccess: async (_data, name) => {
      if (activeName === name) setActiveName(null);
      await invalidate();
    },
  });

  const handleDelete = async (name: string): Promise<void> => {
    setBusy(name);
    setActionError(null);
    setTestResult(null);
    try {
      await deleteMutation.mutateAsync(name);
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

  const listError =
    listQuery.error instanceof Error ? listQuery.error.message : listQuery.isError ? "Failed to list remote workspaces" : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-body text-muted-foreground">
          {targets.length} {targets.length === 1 ? "connection is" : "connections are"} registered.
        </p>
        {writeDeniedReason ? (
          <WorkbenchIconAction label="Add remote workspace" deniedReason={writeDeniedReason}>
            <Plus className="size-3.5" />
          </WorkbenchIconAction>
        ) : (
          <AddRemoteWorkspaceDialog
            trigger={
              <WorkbenchIconAction label="Add remote workspace">
                <Plus className="size-3.5" />
              </WorkbenchIconAction>
            }
            onCreated={() => void invalidate()}
          />
        )}
      </div>
      {listError && <p className="text-body-lg text-status-failed-foreground">{listError}</p>}
      {listQuery.isLoading && targets.length === 0 ? (
        <p className="text-body-lg text-muted-foreground">Loading…</p>
      ) : targets.length === 0 ? (
        <p className="bg-surface/60 px-4 py-8 text-center text-body text-muted-foreground">
          No remote workspaces registered. Add one to mount a workspace hosted on an HPC node.
        </p>
      ) : (
        <ul className="space-y-1">
          {targets.map((t) => {
            const isActive = t.name === activeName;
            return (
              <li
                key={t.name}
                className={`flex flex-col gap-3 px-3 py-3 transition-colors sm:flex-row sm:items-center ${
                  isActive ? "bg-accent-muted/60" : "bg-surface/60 hover:bg-surface"
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-mono text-body font-medium">{t.name}</span>
                    <WorkbenchTag meaning={isActive ? "selection" : "metadata"}>
                      {isActive ? "Active" : "Inactive"}
                    </WorkbenchTag>
                  </div>
                  <div className="mt-1 truncate font-mono text-micro text-muted-foreground">
                    {t.host} → {t.root_path}
                  </div>
                </div>
                <div className="flex flex-none flex-wrap items-center justify-end gap-1">
                  <WorkbenchIconAction
                    label={`Test ${t.name}`}
                    disabled={busy === t.name}
                    deniedReason={writeDeniedReason}
                    onClick={() => void handleTest(t.name)}
                  >
                    <FlaskConical className="size-4" />
                  </WorkbenchIconAction>
                  <WorkbenchIconAction
                    label={isActive ? `${t.name} is active` : `Set ${t.name} active`}
                    disabled={busy === t.name || isActive}
                    deniedReason={writeDeniedReason}
                    onClick={() => void handleSetActive(t.name)}
                  >
                    {isActive ? <Check className="size-4" /> : <Power className="size-4" />}
                  </WorkbenchIconAction>
                  {isActive && (
                    <WorkbenchIconAction
                      label="Re-fetch navigation from remote"
                      disabled={busy === t.name}
                      deniedReason={writeDeniedReason}
                      onClick={() => void handleRefreshCache(t.name)}
                    >
                      <RefreshCw className="h-4 w-4" />
                    </WorkbenchIconAction>
                  )}
                  <WorkbenchIconAction
                    label={`Remove ${t.name}`}
                    kind="ghost"
                    title={isActive ? "Switch to another workspace first" : `Remove ${t.name}`}
                    disabled={busy === t.name || isActive}
                    deniedReason={writeDeniedReason}
                    onClick={() => void handleDelete(t.name)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </WorkbenchIconAction>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {actionError && <p className="text-body-lg text-status-failed-foreground">{actionError}</p>}
      {cacheStatus && (
        <p className="text-label text-muted-foreground">
          Refreshed navigation cache — dropped {cacheStatus.dropped}{" "}
          {cacheStatus.dropped === 1 ? "entry" : "entries"}.
        </p>
      )}
      {openWarnings.length > 0 && (
        <div className="bg-status-warning-soft px-3 py-3 text-body-lg">
          <div className="mb-1 flex items-center gap-2 text-status-warning-foreground">
            <AlertTriangle className="h-4 w-4" />
            <span className="font-medium">
              {openWarnings.length} {openWarnings.length === 1 ? "warning" : "warnings"} while
              fetching the navigation tree
            </span>
          </div>
          <ul className="space-y-1 pl-1 text-label text-muted-foreground">
            {openWarnings.map((w) => (
              <li key={w} className="break-all">
                {w}
              </li>
            ))}
          </ul>
        </div>
      )}
      {testResult && (
        <div className="space-y-1 bg-surface/70 px-3 py-3 text-body-lg">
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
            <p className="text-label text-status-failed-foreground">{testResult.error}</p>
          )}
          <ul className="space-y-1 pl-1">
            {testResult.checks.map((c) => (
              <li key={c.label} className="flex items-start gap-2 text-label text-muted-foreground">
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
