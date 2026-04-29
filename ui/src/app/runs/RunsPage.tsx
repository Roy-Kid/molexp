import { LayoutGrid, ListChecks, RefreshCw } from "lucide-react";
import type { JSX, ReactNode } from "react";
import { Fragment, useCallback, useEffect, useMemo } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { EntityHeader } from "@/app/components/entity";
import type { WorkspaceSnapshot } from "@/app/types";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { formatRelative } from "@/lib/format-time";
import { cn } from "@/lib/utils";
import { ensureLazyPlugin } from "@/plugins/runtime";

import {
  applyFilters,
  computeActivityBuckets,
  computeAvgWaitSeconds,
  computeBackendDistribution,
  computeKpiSparklines,
  computeKpiStats,
  computeTopFailingExperiments,
  type FailingExperimentEntry,
} from "./aggregates";
import { DashboardPanel } from "./DashboardPanel";
import { parseFilterParams, toggleArrayFilter, writeFilterParams } from "./filterParams";
import { RunInspector } from "./inspector/RunInspector";
import { RunsActivityChart } from "./RunsActivityChart";
import { RunsAggregateRow } from "./RunsAggregateRow";
import { RunsGanttChart } from "./RunsGanttChart";
import { RunsJobsTable } from "./RunsJobsTable";
import { RunsKpiStrip } from "./RunsKpiStrip";
import { RunsStatusProgress } from "./RunsStatusProgress";
import { parseRunsTab, type RunsTab, RunsTabBar } from "./RunsTabBar";
import { type GanttMode, RunsTimelineView } from "./RunsTimelineView";
import type { WorkspaceExecutionRow, WorkspaceRunRow, WorkspaceRunsFilters } from "./types";
import { useDashboardLayout } from "./useDashboardLayout";
import { useWorkspaceRuns } from "./useWorkspaceRuns";

interface RunsPageProps {
  snapshot: WorkspaceSnapshot;
}

type DashboardPanelId = "kpi" | "status" | "aggregate" | "activity" | "gantt";

const DASHBOARD_PANEL_IDS: DashboardPanelId[] = ["kpi", "status", "aggregate", "activity", "gantt"];

const DASHBOARD_PANEL_LABELS: Record<DashboardPanelId, string> = {
  kpi: "KPI strip",
  status: "Status mix",
  aggregate: "Backends & failing experiments",
  activity: "Activity chart",
  gantt: "Gantt chart",
};

const DASHBOARD_LAYOUT_STORAGE_KEY = "molexp.runs.dashboard.layout.v2";

const VALID_GANTT_MODES: ReadonlySet<string> = new Set<string>(["runs", "executions"]);

const parseGanttMode = (raw: string | null): GanttMode =>
  raw && VALID_GANTT_MODES.has(raw) ? (raw as GanttMode) : "runs";

const writeRunsParams = (
  prev: URLSearchParams,
  patch: { tab?: RunsTab; runId?: string | null; executionId?: string | null; mode?: GanttMode },
): URLSearchParams => {
  const next = new URLSearchParams(prev);
  if (patch.tab !== undefined) {
    if (patch.tab === "overview") next.delete("tab");
    else next.set("tab", patch.tab);
  }
  if (patch.runId !== undefined) {
    if (patch.runId === null || patch.runId === "") next.delete("runId");
    else next.set("runId", patch.runId);
  }
  if (patch.executionId !== undefined) {
    if (patch.executionId === null || patch.executionId === "") next.delete("executionId");
    else next.set("executionId", patch.executionId);
  }
  if (patch.mode !== undefined) {
    if (patch.mode === "runs") next.delete("mode");
    else next.set("mode", patch.mode);
  }
  return next;
};

export const RunsPage = ({ snapshot: _snapshot }: RunsPageProps): JSX.Element => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const filters = useMemo<WorkspaceRunsFilters>(
    () => parseFilterParams(searchParams),
    [searchParams],
  );

  const tab = parseRunsTab(searchParams.get("tab"));
  const ganttMode = parseGanttMode(searchParams.get("mode"));
  const selectedRunId = searchParams.get("runId");
  const selectedExecutionId = searchParams.get("executionId");

  const layout = useDashboardLayout<DashboardPanelId>(
    DASHBOARD_LAYOUT_STORAGE_KEY,
    DASHBOARD_PANEL_IDS,
  );

  useEffect(() => {
    void ensureLazyPlugin("molq");
  }, []);

  const { rows, truncated, loading, error, lastSyncedAt, refresh } = useWorkspaceRuns();

  const filteredRuns = useMemo(() => applyFilters(rows, filters), [rows, filters]);
  const kpiStats = useMemo(() => computeKpiStats(filteredRuns), [filteredRuns]);
  const avgWait = useMemo(() => computeAvgWaitSeconds(filteredRuns), [filteredRuns]);
  const kpiSparklines = useMemo(() => computeKpiSparklines(filteredRuns), [filteredRuns]);
  const backendDistribution = useMemo(
    () => computeBackendDistribution(filteredRuns),
    [filteredRuns],
  );
  const topFailing = useMemo(() => computeTopFailingExperiments(filteredRuns), [filteredRuns]);
  const activity = useMemo(() => computeActivityBuckets(filteredRuns), [filteredRuns]);

  const selectedRun = useMemo<WorkspaceRunRow | null>(
    () => (selectedRunId ? (rows.find((row) => row.id === selectedRunId) ?? null) : null),
    [rows, selectedRunId],
  );

  const updateFilters = useCallback(
    (next: WorkspaceRunsFilters): void => {
      setSearchParams((prev) => writeFilterParams(prev, next), { replace: true });
    },
    [setSearchParams],
  );

  const setTab = useCallback(
    (next: RunsTab): void => {
      setSearchParams((prev) => writeRunsParams(prev, { tab: next }), { replace: true });
    },
    [setSearchParams],
  );

  const setGanttMode = useCallback(
    (next: GanttMode): void => {
      setSearchParams((prev) => writeRunsParams(prev, { mode: next }), { replace: true });
    },
    [setSearchParams],
  );

  const selectRun = useCallback(
    (run: WorkspaceRunRow): void => {
      setSearchParams((prev) => writeRunsParams(prev, { runId: run.id, executionId: null }), {
        replace: true,
      });
    },
    [setSearchParams],
  );

  const selectExecution = useCallback(
    (run: WorkspaceRunRow, execution: WorkspaceExecutionRow): void => {
      setSearchParams(
        (prev) => writeRunsParams(prev, { runId: run.id, executionId: execution.executionId }),
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setSelectedExecutionId = useCallback(
    (id: string | null): void => {
      setSearchParams((prev) => writeRunsParams(prev, { executionId: id }), { replace: true });
    },
    [setSearchParams],
  );

  const clearSelection = useCallback(() => {
    setSearchParams((prev) => writeRunsParams(prev, { runId: null, executionId: null }), {
      replace: true,
    });
  }, [setSearchParams]);

  const navigateToRun = useCallback(
    (run: WorkspaceRunRow): void => {
      navigate(
        `/projects/${encodeURIComponent(run.projectId)}/experiments/${encodeURIComponent(run.experimentId)}/runs/${encodeURIComponent(run.id)}`,
      );
    },
    [navigate],
  );

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

  const renderPanel = (panelId: DashboardPanelId): ReactNode => {
    switch (panelId) {
      case "kpi":
        return (
          <RunsKpiStrip stats={kpiStats} avgWaitSeconds={avgWait} sparklines={kpiSparklines} />
        );
      case "status":
        return <RunsStatusProgress runs={filteredRuns} onSelectStatus={handleSelectStatus} />;
      case "aggregate":
        return (
          <RunsAggregateRow
            backendDistribution={backendDistribution}
            topFailing={topFailing}
            onSelectBackend={handleSelectBackend}
            onSelectExperiment={handleSelectFailingExperiment}
          />
        );
      case "activity":
        return <RunsActivityChart buckets={activity} />;
      case "gantt":
        return (
          <>
            <RunsGanttChart
              rows={filteredRuns}
              mode={ganttMode}
              onSelectRun={selectRun}
              onSelectExecution={selectExecution}
            />
            <p className="mt-2 text-[11px] italic text-muted-foreground">
              Click a bar to load it in the inspector. Faded bars are queued / pending.
            </p>
          </>
        );
    }
  };

  const headerSummary = truncated
    ? `Showing first ${rows.length} runs (truncated). Narrow filters or raise the limit.`
    : `${filteredRuns.length} of ${rows.length} runs match current filters.`;

  return (
    <div className="flex h-full min-h-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <EntityHeader
          icon={ListChecks}
          title="Runs"
          subtitle={headerSummary}
          actions={
            <>
              {tab === "overview" && (
                <PanelManager
                  allIds={DASHBOARD_PANEL_IDS}
                  hiddenIds={layout.hiddenIds}
                  onToggle={layout.toggleVisibility}
                  onReset={layout.reset}
                />
              )}
              <span className="text-xs text-muted-foreground">
                Last synced {lastSyncedAt ? formatRelative(lastSyncedAt.toISOString()) : "—"}
              </span>
              <Button
                type="button"
                size="icon"
                variant="ghost"
                onClick={refresh}
                disabled={loading}
                aria-label={loading ? "Refreshing" : "Refresh"}
                title={loading ? "Refreshing…" : "Refresh"}
                className="h-7 w-7 text-muted-foreground hover:text-foreground"
              >
                <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
              </Button>
            </>
          }
        />
        <div className="border-b border-border bg-background px-4 md:px-6">
          <RunsTabBar value={tab} onChange={setTab} />
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-4 md:p-6">
          {error && (
            <div className="mb-4 rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}

          {tab === "overview" && (
            <div className="space-y-4">
              {layout.rows.map((row) => (
                <DashboardRowView
                  key={row.id}
                  rowId={row.id}
                  panels={row.panels}
                  renderPanel={renderPanel}
                  labels={DASHBOARD_PANEL_LABELS}
                  onReorder={layout.reorder}
                  onRemove={layout.hide}
                />
              ))}
              {layout.rows.length === 0 && (
                <div className="rounded border border-dashed border-border p-6 text-center text-xs text-muted-foreground">
                  All panels hidden. Use the layout menu above to restore them.
                </div>
              )}
            </div>
          )}

          {tab === "jobs" && (
            <RunsJobsTable
              rows={filteredRuns}
              selectedRunId={selectedRunId}
              onSelectRun={selectRun}
            />
          )}

          {tab === "timeline" && (
            <RunsTimelineView
              rows={filteredRuns}
              mode={ganttMode}
              onModeChange={setGanttMode}
              onSelectRun={selectRun}
              onSelectExecution={selectExecution}
            />
          )}
        </div>
      </div>

      <RunInspector
        run={selectedRun}
        selectedExecutionId={selectedExecutionId}
        onSelectExecution={setSelectedExecutionId}
        onClear={clearSelection}
        onOpenRun={navigateToRun}
      />
    </div>
  );
};

interface PanelManagerProps {
  allIds: DashboardPanelId[];
  hiddenIds: DashboardPanelId[];
  onToggle: (id: string) => void;
  onReset: () => void;
}

const PanelManager = ({ allIds, hiddenIds, onToggle, onReset }: PanelManagerProps): JSX.Element => {
  const hiddenSet = new Set(hiddenIds);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size="sm" variant="outline" title="Layout">
          <LayoutGrid className="mr-1.5 h-3.5 w-3.5" />
          Layout
          {hiddenIds.length > 0 && (
            <span className="ml-1 rounded-full bg-muted px-1.5 text-[10px] font-medium tabular-nums text-muted-foreground">
              {hiddenIds.length}
            </span>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>Panels</DropdownMenuLabel>
        {allIds.map((id) => (
          <DropdownMenuCheckboxItem
            key={id}
            checked={!hiddenSet.has(id)}
            onCheckedChange={() => onToggle(id)}
            onSelect={(event) => event.preventDefault()}
          >
            {DASHBOARD_PANEL_LABELS[id]}
          </DropdownMenuCheckboxItem>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => onReset()}>Reset layout</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

interface DashboardRowViewProps {
  rowId: string;
  panels: DashboardPanelId[];
  renderPanel: (id: DashboardPanelId) => ReactNode;
  labels: Record<DashboardPanelId, string>;
  onReorder: ReturnType<typeof useDashboardLayout<DashboardPanelId>>["reorder"];
  onRemove: ReturnType<typeof useDashboardLayout<DashboardPanelId>>["hide"];
}

const DashboardRowView = ({
  rowId,
  panels,
  renderPanel,
  labels,
  onReorder,
  onRemove,
}: DashboardRowViewProps): JSX.Element => {
  if (panels.length === 1) {
    const panelId = panels[0];
    return (
      <DashboardPanel
        id={panelId}
        title={labels[panelId]}
        onReorder={onReorder}
        onRemove={onRemove}
      >
        {renderPanel(panelId)}
      </DashboardPanel>
    );
  }

  return (
    <ResizablePanelGroup
      direction="horizontal"
      autoSaveId={`molexp.runs.dashboard.row.${rowId}`}
      className="!h-auto min-h-[180px] gap-2"
    >
      {panels.map((panelId, idx) => (
        <Fragment key={panelId}>
          {idx > 0 && (
            <ResizableHandle withHandle className="!w-1 bg-transparent hover:bg-border/60" />
          )}
          <ResizablePanel defaultSize={100 / panels.length} minSize={15} className="min-w-0">
            <DashboardPanel
              id={panelId}
              title={labels[panelId]}
              onReorder={onReorder}
              onRemove={onRemove}
            >
              {renderPanel(panelId)}
            </DashboardPanel>
          </ResizablePanel>
        </Fragment>
      ))}
    </ResizablePanelGroup>
  );
};
