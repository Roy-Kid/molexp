/**
 * Bodies for the workbench BottomPanel tabs, scoped by current selection.
 */

import type { JSX } from "react";
import { useEffect, useMemo, useState } from "react";

import { workspaceApi } from "@/app/state/api";
import { pulseSync } from "@/app/state/syncPulse";
import type { Selection, WorkspaceSnapshot } from "@/app/types";
import { RunStatusBadge, WorkbenchAction, WorkbenchOperationState } from "@/components/workbench";
import { formatRelative } from "@/lib/format-time";
import { cn } from "@/lib/utils";

interface BottomPanelContentProps {
  selection: Selection | null;
  snapshot: WorkspaceSnapshot;
  onSelectRun: (runId: string) => void;
}

const resolveRunScope = (
  selection: Selection | null,
  snapshot: WorkspaceSnapshot,
): {
  projectId: string;
  experimentId: string;
  runId: string;
  status: string | null;
} | null => {
  if (!selection) return null;
  if (selection.objectType === "run") {
    const run = snapshot.runs.find((r) => r.id === selection.objectId);
    if (!run) return null;
    return {
      projectId: run.projectId,
      experimentId: run.experimentId,
      runId: run.id,
      status: run.status ?? null,
    };
  }
  if (selection.objectType === "task") {
    const run = snapshot.runs.find((r) => r.id === selection.runId);
    if (!run) return null;
    return {
      projectId: run.projectId,
      experimentId: run.experimentId,
      runId: run.id,
      status: run.status ?? null,
    };
  }
  return null;
};

export const LogsSlot = ({
  selection,
  snapshot,
}: Pick<BottomPanelContentProps, "selection" | "snapshot">): JSX.Element => {
  const scope = resolveRunScope(selection, snapshot);
  const [stdout, setStdout] = useState<string | null>(null);
  const [stderr, setStderr] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0);
  const projectId = scope?.projectId;
  const experimentId = scope?.experimentId;
  const runId = scope?.runId;

  useEffect(() => {
    void tick;
    if (!projectId || !experimentId || !runId) {
      setStdout(null);
      setStderr(null);
      setError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    workspaceApi
      .getRunLogs(projectId, experimentId, runId)
      .then((response) => {
        if (cancelled) return;
        setStdout(response.stdout ?? null);
        setStderr(response.stderr ?? null);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(String(err));
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
          pulseSync();
        }
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, experimentId, runId, tick]);

  if (!scope) {
    return (
      <WorkbenchOperationState
        kind="empty"
        density="compact"
        title="No logs in scope"
        detail="Select a run to stream stdout and stderr here."
      />
    );
  }

  if (loading && !stdout && !stderr && !error) {
    return (
      <WorkbenchOperationState
        kind="loading"
        density="compact"
        title="Loading logs…"
        skeletonRows={4}
      />
    );
  }

  if (error) {
    return (
      <WorkbenchOperationState
        kind="error"
        density="compact"
        title="Could not load logs"
        detail={error}
        action={
          <WorkbenchAction kind="secondary" size="compact" onClick={() => setTick((t) => t + 1)}>
            Retry
          </WorkbenchAction>
        }
      />
    );
  }

  const body = (
    <div className="flex h-full min-h-0 flex-col font-mono text-micro">
      <div className="flex flex-none items-center gap-2 border-b border-border/70 bg-muted/30 px-3 py-1 text-muted-foreground">
        <span>stdout/stderr</span>
        <span className="text-muted-foreground/50">·</span>
        <span className="text-foreground tabular-nums">{scope.runId}</span>
        {scope.status && <RunStatusBadge status={scope.status} size="sm" />}
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-auto p-3">
        <section>
          <div className="mb-1 text-micro uppercase tracking-wide text-muted-foreground">
            stdout
          </div>
          <pre className="whitespace-pre-wrap text-foreground">
            {stdout || "No stdout captured."}
          </pre>
        </section>
        <section>
          <div className="mb-1 text-micro uppercase tracking-wide text-muted-foreground">
            stderr
          </div>
          <pre className="whitespace-pre-wrap text-status-failed-foreground">
            {stderr || "No stderr captured."}
          </pre>
        </section>
      </div>
    </div>
  );

  if (scope.status === "running") {
    return (
      <WorkbenchOperationState kind="running" title="Run in progress" detail={scope.runId}>
        {body}
      </WorkbenchOperationState>
    );
  }

  return body;
};

export const RunsSlot = ({
  snapshot,
  onSelectRun,
}: Pick<BottomPanelContentProps, "snapshot" | "onSelectRun">): JSX.Element => {
  const rows = useMemo(() => {
    return [...snapshot.runs]
      .sort((a, b) => (b.updatedAt ?? "").localeCompare(a.updatedAt ?? ""))
      .slice(0, 40);
  }, [snapshot.runs]);

  if (rows.length === 0) {
    return (
      <WorkbenchOperationState
        kind="empty"
        density="compact"
        title="No runs in this workspace yet"
        detail="Start a run from an experiment to see it here."
      />
    );
  }

  return (
    <ul className="divide-y divide-border/60">
      {rows.map((run) => (
        <li key={run.id}>
          <button
            type="button"
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-body transition-colors duration-[var(--motion-fast)] ease-[var(--motion-ease)] hover:bg-interactive/50"
            onClick={() => onSelectRun(run.id)}
          >
            <RunStatusBadge status={run.status} showLabel={false} />
            <span className="min-w-0 flex-1 truncate font-mono text-label tabular-nums">
              {run.id}
            </span>
            <span className="truncate text-micro text-muted-foreground">{run.experimentId}</span>
            <span className="flex-none text-micro text-muted-foreground tabular-nums">
              {formatRelative(run.updatedAt)}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
};

export const ArtifactsSlot = ({
  selection,
  snapshot,
}: Pick<BottomPanelContentProps, "selection" | "snapshot">): JSX.Element => {
  const scope = resolveRunScope(selection, snapshot);
  const assets = useMemo(() => {
    if (!scope) return [];
    return snapshot.assets.filter((a) => a.runId === scope.runId).slice(0, 50);
  }, [scope, snapshot.assets]);

  if (!scope) {
    return (
      <WorkbenchOperationState
        kind="empty"
        density="compact"
        title="No artifacts in scope"
        detail="Select a run to browse its artifacts without leaving the graph."
      />
    );
  }

  if (assets.length === 0) {
    return (
      <WorkbenchOperationState
        kind="empty"
        density="compact"
        title="No artifacts yet"
        detail={`Nothing registered for run ${scope.runId}.`}
      />
    );
  }

  return (
    <ul className="divide-y divide-border/60">
      {assets.map((asset) => (
        <li key={asset.id} className={cn("flex items-center gap-2 px-3 py-2 text-body")}>
          <span className="min-w-0 flex-1 truncate font-mono text-label">
            {asset.name ?? asset.id}
          </span>
          <span className="flex-none text-micro text-muted-foreground">{asset.kind}</span>
        </li>
      ))}
    </ul>
  );
};

export const ProblemsSlot = (): JSX.Element => (
  <WorkbenchOperationState
    kind="empty"
    density="compact"
    title="No problems"
    detail="Validation and compile issues for the active graph will list here."
  />
);
