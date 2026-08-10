import { LayoutGrid, ListChecks, RefreshCw } from "lucide-react";
import type { JSX, ReactNode } from "react";
import { Fragment, useCallback, useEffect, useMemo } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { DashboardCard, EmptyState, EntityHeader } from "@/app/components/entity";
import { runPath } from "@/app/entities/paths";
import type { WorkspaceSnapshot } from "@/app/types";
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
import { WorkbenchIconAction } from "@/components/workbench";
import { formatRelative } from "@/lib/format-time";
import { cn } from "@/lib/utils";
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
import type { RunInspectorRegistration } from "./inspector/RunInspector";
import {
  DEFAULT_JOBS_SORT,
  DEFAULT_PAGE_SIZE,
  formatJobsSort,
  type JobsSort,
  parseJobsSort,
  parsePage,
  parsePageSize,
} from "./jobsTable";
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
import { WorkspaceActivityFeed } from "./WorkspaceActivityFeed";

interface RunsPageProps {
  snapshot: WorkspaceSnapshot;
  onInspectorChange: (registration: RunInspectorRegistration | null) => void;
}

type DashboardPanelId = "kpi" | "status" | "aggregate" | "activity" | "feed" | "gantt";

const DASHBOARD_PANEL_IDS: DashboardPanelId[] = [
  "kpi",
  "status",
  "aggregate",
  "activity",
  "feed",
  "gantt",
];

const DASHBOARD_PANEL_LABELS: Record<DashboardPanelId, string> = {
  kpi: "Key metrics",
  status: "Status mix",
  aggregate: "Backends & failures",
  activity: "Activity",
  feed: "Workspace activity",
  gantt: "Gantt",
};

const DASHBOARD_PANEL_DESCRIPTIONS: Partial<Record<DashboardPanelId, string>> = {
  status: "Click a segment to filter",
  aggregate: "Backend load and experiments with the most failures",
  activity: "Last 24 hours",
  feed: "Recent workspace events",
  gantt: "Click a bar to inspect",
};

const DASHBOARD_LAYOUT_STORAGE_KEY = "molexp.runs.dashboard.layout.v2";

const VALID_GANTT_MODES: ReadonlySet<string> = new Set<string>(["runs", "executions"]);

const parseGanttMode = (raw: string | null): GanttMode =>
  raw && VALID_GANTT_MODES.has(raw) ? (raw as GanttMode) : "runs";

const writeRunsParams = (
  prev: URLSearchParams,
  patch: {
    tab?: RunsTab;
    runId?: string | null;
    executionId?: string | null;
    mode?: GanttMode;
    sort?: JobsSort | null;
    page?: number | null;
    pageSize?: number | null;
  },
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
  if (patch.sort !== undefined) {
    if (
      patch.sort === null ||
      (patch.sort.key === DEFAULT_JOBS_SORT.key && patch.sort.dir === DEFAULT_JOBS_SORT.dir)
    ) {
      next.delete("sort");
    } else {
      next.set("sort", formatJobsSort(patch.sort));
    }
  }
  if (patch.page !== undefined) {
    if (patch.page === null || patch.page <= 1) next.delete("page");
    else next.set("page", String(patch.page));
  }
  if (patch.pageSize !== undefined) {
    if (patch.pageSize === null || patch.pageSize === DEFAULT_PAGE_SIZE) next.delete("pageSize");
    else next.set("pageSize", String(patch.pageSize));
  }
  return next;
};

export const RunsPage = ({
  snapshot: _snapshot,
  onInspectorChange,
}: RunsPageProps): JSX.Element => {
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
  const jobsSort = useMemo(() => parseJobsSort(searchParams.get("sort")), [searchParams]);
  const jobsPage = useMemo(() => parsePage(searchParams.get("page")), [searchParams]);
  const jobsPageSize = useMemo(() => parsePageSize(searchParams.get("pageSize")), [searchParams]);

  const layout = useDashboardLayout<DashboardPanelId>(
    DASHBOARD_LAYOUT_STORAGE_KEY,
    DASHBOARD_PANEL_IDS,
  );

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

  const setJobsSort = useCallback(
    (next: JobsSort): void => {
      setSearchParams((prev) => writeRunsParams(prev, { sort: next, page: 1 }), { replace: true });
    },
    [setSearchParams],
  );

  const setJobsPage = useCallback(
    (next: number): void => {
      setSearchParams((prev) => writeRunsParams(prev, { page: next }), { replace: true });
    },
    [setSearchParams],
  );

  const setJobsPageSize = useCallback(
    (next: number): void => {
      setSearchParams((prev) => writeRunsParams(prev, { pageSize: next, page: 1 }), {
        replace: true,
      });
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

  // Feed link targets (vision-loop-12): the activity feed hands back plain
  // ids/paths, not row objects.
  const knownRunIds = useMemo(() => new Set(rows.map((row) => row.id)), [rows]);
  const selectRunById = useCallback(
    (runId: string): void => {
      setSearchParams((prev) => writeRunsParams(prev, { runId, executionId: null }), {
        replace: true,
      });
    },
    [setSearchParams],
  );
  const openKnowledge = useCallback(
    (path: string): void => {
      navigate(`/knowledge/${path.split("/").map(encodeURIComponent).join("/")}`);
    },
    [navigate],
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
      navigate(runPath(run.projectId, run.experimentId, run.id));
    },
    [navigate],
  );

  useEffect(() => {
    if (!selectedRun) {
      onInspectorChange(null);
      return;
    }

    onInspectorChange({
      run: selectedRun,
      selectedExecutionId,
      onSelectExecution: setSelectedExecutionId,
      onClear: clearSelection,
      onOpenRun: navigateToRun,
    });
  }, [
    clearSelection,
    navigateToRun,
    onInspectorChange,
    selectedExecutionId,
    selectedRun,
    setSelectedExecutionId,
  ]);

  useEffect(
    () => () => {
      onInspectorChange(null);
    },
    [onInspectorChange],
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
      case "feed":
        return (
          <WorkspaceActivityFeed
            knownRunIds={knownRunIds}
            onSelectRun={selectRunById}
            onOpenKnowledge={openKnowledge}
          />
        );
      case "gantt":
        return (
          <RunsGanttChart
            rows={filteredRuns}
            mode={ganttMode}
            onSelectRun={selectRun}
            onSelectExecution={selectExecution}
          />
        );
    }
  };

  const headerSummary = truncated
    ? `Showing first ${rows.length} runs (truncated). Narrow filters or raise the limit.`
    : `${filteredRuns.length} of ${rows.length} runs match current filters`;

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col overflow-hidden">
      <EntityHeader
        icon={ListChecks}
        title="Runs"
        titleTooltip={headerSummary}
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
            <WorkbenchIconAction
              label={loading ? "Refreshing runs" : "Refresh runs"}
              kind="ghost"
              type="button"
              onClick={refresh}
              disabled={loading}
              title={
                loading
                  ? "Refreshing…"
                  : lastSyncedAt
                    ? `Refresh · synced ${formatRelative(lastSyncedAt.toISOString())}`
                    : "Refresh"
              }
              size="default"
              className="text-muted-foreground hover:text-foreground"
            >
              <RefreshCw className={cn("h-3.5 w-3.5", loading && "mol-motion-progress-spin")} />
            </WorkbenchIconAction>
          </>
        }
      />
      <div className="shrink-0 border-b border-border/60 bg-background px-4">
        <RunsTabBar value={tab} onChange={setTab} />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4 md:p-4">
        {error && (
          <DashboardCard title="Could not load runs" variant="destructive" className="mb-4">
            <p className="text-body-lg text-destructive">{error}</p>
          </DashboardCard>
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
                descriptions={DASHBOARD_PANEL_DESCRIPTIONS}
                onReorder={layout.reorder}
                onRemove={layout.hide}
              />
            ))}
            {layout.rows.length === 0 && (
              <div className="flex min-h-48 items-center justify-center border-y border-dashed border-border/70">
                <EmptyState
                  density="compact"
                  icon={<LayoutGrid className="h-5 w-5" />}
                  title="All panels hidden"
                  description="Use Layout in the header to restore the dashboard panels."
                />
              </div>
            )}
          </div>
        )}

        {tab === "jobs" && (
          <RunsJobsTable
            rows={filteredRuns}
            selectedRunId={selectedRunId}
            onSelectRun={selectRun}
            sort={jobsSort}
            onSortChange={setJobsSort}
            page={jobsPage}
            pageSize={jobsPageSize}
            onPageChange={setJobsPage}
            onPageSizeChange={setJobsPageSize}
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
        <WorkbenchIconAction
          label={hiddenIds.length > 0 ? `Layout, ${hiddenIds.length} panels hidden` : "Layout"}
          className="relative"
        >
          <LayoutGrid className="h-3.5 w-3.5" />
          {hiddenIds.length > 0 && (
            <span className="absolute -right-1 -top-1 min-w-4 rounded-full bg-accent px-1 text-micro font-medium tabular-nums text-accent-foreground">
              {hiddenIds.length}
            </span>
          )}
        </WorkbenchIconAction>
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
  descriptions: Partial<Record<DashboardPanelId, string>>;
  onReorder: ReturnType<typeof useDashboardLayout<DashboardPanelId>>["reorder"];
  onRemove: ReturnType<typeof useDashboardLayout<DashboardPanelId>>["hide"];
}

const DashboardRowView = ({
  rowId,
  panels,
  renderPanel,
  labels,
  descriptions,
  onReorder,
  onRemove,
}: DashboardRowViewProps): JSX.Element => {
  if (panels.length === 1) {
    const panelId = panels[0];
    return (
      <DashboardPanel
        id={panelId}
        title={labels[panelId]}
        description={descriptions[panelId]}
        bare={panelId === "kpi"}
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
      autoSavePanelIds={panels}
      className="!h-auto min-h-44 gap-2"
    >
      {panels.map((panelId, idx) => (
        <Fragment key={panelId}>
          {idx > 0 && (
            <ResizableHandle withHandle className="!w-1 bg-transparent hover:bg-border/60" />
          )}
          <ResizablePanel
            id={panelId}
            defaultSize={100 / panels.length}
            minSize={15}
            className="min-w-0"
          >
            <DashboardPanel
              id={panelId}
              title={labels[panelId]}
              description={descriptions[panelId]}
              bare={panelId === "kpi"}
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
