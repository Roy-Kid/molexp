/**
 * Workspace Copilot — read-only summary + advisory next-actions.
 *
 * close-loop-04: consumes GET /api/workspace/copilot only. Never executes
 * lifecycle verbs; high-risk actions show a "needs approval" badge.
 */

import type { JSX } from "react";
import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { NextActionResponse } from "@/api/generated/models/NextActionResponse";
import type { WorkspaceSummaryResponse } from "@/api/generated/models/WorkspaceSummaryResponse";
import { WorkspaceService } from "@/api/generated/services/WorkspaceService";
import { pathForNextAction } from "@/app/components/copilotPaths";
import type { WorkspaceSnapshot } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { toast } from "@/components/ui/toast";
import { WorkbenchAction } from "@/components/workbench";

export { pathForNextAction } from "@/app/components/copilotPaths";

interface CopilotPanelProps {
  snapshot: WorkspaceSnapshot;
  /** Optional external trigger to re-fetch (e.g. after diagnose). */
  refreshKey?: number;
}

const ActionRow = ({
  action,
  snapshot,
  onDiagnosed,
}: {
  action: NextActionResponse;
  snapshot: WorkspaceSnapshot;
  onDiagnosed?: () => void;
}): JSX.Element => {
  const navigate = useNavigate();
  const path = pathForNextAction(action, snapshot);
  const [busy, setBusy] = useState(false);

  const onClick = async (): Promise<void> => {
    if (action.kind === "diagnose_failed_run") {
      const run = snapshot.runs.find((r) => r.id === action.target);
      if (!run) {
        if (path) navigate(path);
        return;
      }
      const ok = window.confirm(
        `Analyze failure for run ${run.id}? Writes a FailureAnalysis knowledge item.`,
      );
      if (!ok) return;
      setBusy(true);
      try {
        const { postAnalyzeFailure } = await import("@/app/runs/analyzeFailure");
        const result = await postAnalyzeFailure(run.projectId, run.experimentId, run.id);
        toast.success(result.path ? `Analyzed → ${result.path}` : "Failure analyzed");
        onDiagnosed?.();
        if (path) navigate(path);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
      return;
    }
    // retry_failed_run and others: navigate only — never auto-resume.
    if (path) navigate(path);
  };

  return (
    <WorkbenchAction
      kind="ghost"
      size="content"
      type="button"
      disabled={!path || busy}
      onClick={() => {
        void onClick();
      }}
      className="group flex w-full flex-col items-start gap-1 rounded-control px-2 py-2 text-left transition-colors hover:bg-muted/60 disabled:opacity-40"
    >
      <div className="flex w-full items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-label font-medium text-foreground">
          {action.kind.replace(/_/g, " ")}
        </span>
        {action.requiresProposal ? (
          <Badge variant="outline" className="text-micro shrink-0">
            needs approval
          </Badge>
        ) : null}
      </div>
      <span className="w-full text-micro text-muted-foreground line-clamp-2">
        {action.rationale}
      </span>
    </WorkbenchAction>
  );
};

export const CopilotPanel = ({ snapshot, refreshKey = 0 }: CopilotPanelProps): JSX.Element => {
  const [data, setData] = useState<WorkspaceSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [localRefresh, setLocalRefresh] = useState(0);
  const effectiveRefresh = refreshKey + localRefresh;

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    WorkspaceService.getWorkspaceCopilotApiWorkspaceCopilotGet()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    void effectiveRefresh;
    load();
  }, [load, effectiveRefresh]);

  if (loading && !data) {
    return (
      <section className="px-3 py-3" aria-busy="true">
        <h3 className="text-micro font-semibold uppercase tracking-wide text-muted-foreground">
          Copilot
        </h3>
        <p className="mt-2 text-label text-muted-foreground">Loading workspace summary…</p>
      </section>
    );
  }

  if (error && !data) {
    return (
      <section className="px-3 py-3">
        <h3 className="text-micro font-semibold uppercase tracking-wide text-muted-foreground">
          Copilot
        </h3>
        <p className="mt-2 text-label text-destructive" role="alert">
          {error}
        </p>
        <WorkbenchAction kind="ghost" size="content" type="button" onClick={load} className="mt-2">
          Retry
        </WorkbenchAction>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="px-3 py-3">
        <h3 className="text-micro font-semibold uppercase tracking-wide text-muted-foreground">
          Copilot
        </h3>
        <p className="mt-2 text-label text-muted-foreground">
          No summary yet. Open a workspace and refresh.
        </p>
      </section>
    );
  }

  const actions = data.nextActions ?? [];
  const failed = data.failedRuns ?? [];

  return (
    <section className="border-b border-border/70 px-2 py-3">
      <div className="flex items-center justify-between px-2 pb-2">
        <h3 className="text-micro font-semibold uppercase tracking-wide text-muted-foreground">
          Copilot
        </h3>
        <WorkbenchAction
          kind="ghost"
          size="content"
          type="button"
          onClick={load}
          className="text-micro text-muted-foreground"
        >
          Refresh
        </WorkbenchAction>
      </div>
      <p className="px-2 pb-2 text-label text-foreground">{data.headline}</p>
      <div className="flex flex-wrap gap-2 px-2 pb-2 text-micro tabular-nums text-muted-foreground">
        {Object.entries(data.counts ?? {}).map(([key, value]) => (
          <span key={key}>
            {key.replace(/_/g, " ")}: {value}
          </span>
        ))}
      </div>
      {failed.length > 0 ? (
        <p className="px-2 pb-1 text-micro text-muted-foreground">
          {failed.length} failed run{failed.length === 1 ? "" : "s"} in summary
        </p>
      ) : null}
      <div className="space-y-1">
        {actions.length === 0 ? (
          <p className="px-2 py-2 text-label text-muted-foreground">
            No next actions — workspace health looks calm.
          </p>
        ) : (
          actions.map((action) => (
            <ActionRow
              key={`${action.kind}:${action.target}:${action.rationale}`}
              action={action}
              snapshot={snapshot}
              onDiagnosed={() => setLocalRefresh((n) => n + 1)}
            />
          ))
        )}
      </div>
    </section>
  );
};
