import { RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { JSX } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { ensureLazyPlugin } from "@/plugins/runtime";
import { formatRelative } from "@/lib/format-time";
import { cn } from "@/lib/utils";
import type { WorkspaceSnapshot } from "@/app/types";

import {
  applyFilters,
  computeActivityBuckets,
  computeAvgWaitSeconds,
  computeBackendDistribution,
  computeKpiStats,
  computeTopFailingExperiments,
  type FailingExperimentEntry,
} from "./aggregates";
import { ExecutionDetailDrawer } from "./ExecutionDetailDrawer";
import { parseFilterParams, toggleArrayFilter, writeFilterParams } from "./filterParams";
import { RunDetailDrawer } from "./RunDetailDrawer";
import { RunsActivityChart } from "./RunsActivityChart";
import { RunsAggregateRow } from "./RunsAggregateRow";
import { RunsGanttChart } from "./RunsGanttChart";
import { RunsKpiStrip } from "./RunsKpiStrip";
import { RunsStatusProgress } from "./RunsStatusProgress";
import { useWorkspaceRuns } from "./useWorkspaceRuns";
import type {
  WorkspaceExecutionRow,
  WorkspaceRunRow,
  WorkspaceRunsFilters,
} from "./types";

interface RunsPageProps {
  snapshot: WorkspaceSnapshot;
}

interface DrawerSelection {
  runId: string;
  executionId: string | null;
}

type GanttMode = "runs" | "executions";

export const RunsPage = ({ snapshot: _snapshot }: RunsPageProps): JSX.Element => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const filters = useMemo<WorkspaceRunsFilters>(
    () => parseFilterParams(searchParams),
    [searchParams],
  );
  const [ganttMode, setGanttMode] = useState<GanttMode>("runs");
  const [drawerSelection, setDrawerSelection] = useState<DrawerSelection | null>(null);
  const initialAutoSelectRunId = useRef<string | null>(searchParams.get("runId"));

  useEffect(() => {
    void ensureLazyPlugin("molq");
  }, []);

  const { rows, truncated, loading, error, lastSyncedAt, refresh } = useWorkspaceRuns();

  const filteredRuns = useMemo(() => applyFilters(rows, filters), [rows, filters]);
  const kpiStats = useMemo(() => computeKpiStats(filteredRuns), [filteredRuns]);
  const avgWait = useMemo(() => computeAvgWaitSeconds(filteredRuns), [filteredRuns]);
  const backendDistribution = useMemo(
    () => computeBackendDistribution(filteredRuns),
    [filteredRuns],
  );
  const topFailing = useMemo(() => computeTopFailingExperiments(filteredRuns), [filteredRuns]);
  const activity = useMemo(() => computeActivityBuckets(filteredRuns), [filteredRuns]);

  useEffect(() => {
    const runId = initialAutoSelectRunId.current;
    if (!runId) return;
    const found = rows.find((row) => row.id === runId);
    if (!found) return;
    initialAutoSelectRunId.current = null;
    setDrawerSelection({
      runId,
      executionId: found.executions[0]?.executionId ?? null,
    });
  }, [rows]);

  const drawerRun = useMemo<WorkspaceRunRow | null>(
    () => rows.find((row) => row.id === drawerSelection?.runId) ?? null,
    [rows, drawerSelection],
  );
  const drawerExecution = useMemo<WorkspaceExecutionRow | null>(() => {
    if (!drawerRun || !drawerSelection?.executionId) return null;
    return (
      drawerRun.executions.find((exec) => exec.executionId === drawerSelection.executionId) ?? null
    );
  }, [drawerRun, drawerSelection]);

  const updateFilters = (next: WorkspaceRunsFilters): void => {
    setSearchParams((prev) => writeFilterParams(prev, next), { replace: true });
  };

  const openRunDrawer = (run: WorkspaceRunRow): void => {
    setDrawerSelection({ runId: run.id, executionId: null });
  };

  const openExecutionDrawer = (run: WorkspaceRunRow, execution: WorkspaceExecutionRow): void => {
    setDrawerSelection({ runId: run.id, executionId: execution.executionId });
  };

  const navigateToRun = (run: WorkspaceRunRow): void => {
    navigate(
      `/projects/${encodeURIComponent(run.projectId)}/experiments/${encodeURIComponent(run.experimentId)}/runs/${encodeURIComponent(run.id)}`,
    );
  };

  const handleSelectBackend = (backend: string): void => {
    updateFilters(toggleArrayFilter(filters, "backend", backend));
  };

  const handleSelectFailingExperiment = (entry: FailingExperimentEntry): void => {
    let next = filters;
    if (!next.projectId?.includes(entry.projectId)) {
      next = toggleArrayFilter(next, "projectId", entry.projectId);
    }
    if (!next.experimentId?.includes(entry.experimentId)) {
      next = toggleArrayFilter(next, "experimentId", entry.experimentId);
    }
    updateFilters(next);
  };

  const handleSelectStatus = (status: string): void => {
    updateFilters(toggleArrayFilter(filters, "status", status));
  };

  return (
    <div className="flex h-full min-h-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col gap-4 overflow-y-auto p-4 md:p-6">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold">Runs operations dashboard</h2>
            <p className="text-xs text-muted-foreground">
              {truncated
                ? `Showing first ${rows.length} runs (truncated). Narrow filters or raise the limit.`
                : `${filteredRuns.length} of ${rows.length} runs match current filters.`}
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <ModeToggle value={ganttMode} onChange={setGanttMode} />
            <span>
              Last synced{" "}
              {lastSyncedAt ? formatRelative(lastSyncedAt.toISOString()) : "—"}
            </span>
            <Button size="sm" variant="outline" onClick={refresh} disabled={loading}>
              <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </header>

        <RunsKpiStrip stats={kpiStats} avgWaitSeconds={avgWait} />

        <RunsStatusProgress runs={filteredRuns} onSelectStatus={handleSelectStatus} />

        <RunsAggregateRow
          backendDistribution={backendDistribution}
          topFailing={topFailing}
          onSelectBackend={handleSelectBackend}
          onSelectExperiment={handleSelectFailingExperiment}
        />

        <RunsActivityChart buckets={activity} />

        {error && (
          <div className="rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        )}

        <RunsGanttChart
          rows={filteredRuns}
          mode={ganttMode}
          onSelectRun={openRunDrawer}
          onSelectExecution={openExecutionDrawer}
        />

        <p className="text-[11px] italic text-muted-foreground">
          Click a bar to open its detail drawer. Faded bars are queued / pending.
        </p>
      </div>

      {drawerRun && drawerExecution && (
        <ExecutionDetailDrawer
          run={drawerRun}
          execution={drawerExecution}
          onClose={() => setDrawerSelection(null)}
        />
      )}
      {drawerRun && !drawerExecution && (
        <RunDetailDrawer
          run={drawerRun}
          onClose={() => setDrawerSelection(null)}
          onOpenRun={() => navigateToRun(drawerRun)}
          onSelectExecution={(exec) => openExecutionDrawer(drawerRun, exec)}
        />
      )}
    </div>
  );
};

interface ModeToggleProps {
  value: GanttMode;
  onChange: (next: GanttMode) => void;
}

const ModeToggle = ({ value, onChange }: ModeToggleProps): JSX.Element => (
  <div className="flex items-center overflow-hidden rounded border border-border">
    {(["runs", "executions"] as const).map((mode) => (
      <button
        key={mode}
        type="button"
        onClick={() => onChange(mode)}
        className={cn(
          "px-2 py-1 text-[11px] uppercase tracking-wide transition-colors",
          value === mode
            ? "bg-accent text-accent-foreground"
            : "text-muted-foreground hover:bg-muted/40",
        )}
      >
        By {mode}
      </button>
    ))}
  </div>
);
