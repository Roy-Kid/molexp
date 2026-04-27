import { useMemo } from "react";
import type { JSX } from "react";

import { Plot } from "@/lib/plot";

import type { WorkspaceExecutionRow, WorkspaceRunRow } from "./types";

interface RunsGanttChartProps {
  rows: WorkspaceRunRow[];
  mode: "runs" | "executions";
  onSelectRun: (run: WorkspaceRunRow) => void;
  onSelectExecution: (run: WorkspaceRunRow, execution: WorkspaceExecutionRow) => void;
}

interface GanttTask {
  runId: string;
  executionId: string | null;
  label: string;
  start: Date;
  end: Date;
  statusGroup: StatusGroup;
  statusRaw: string;
  hover: string;
  isPending: boolean;
}

type StatusGroup = "running" | "succeeded" | "failed" | "pending" | "cancelled";

const STATUS_GROUP_COLOR: Record<StatusGroup, string> = {
  running: "#3b82f6",
  succeeded: "#10b981",
  failed: "#ef4444",
  pending: "#a3a3a3",
  cancelled: "#71717a",
};

const STATUS_GROUP_LABEL: Record<StatusGroup, string> = {
  running: "Running",
  succeeded: "Succeeded",
  failed: "Failed",
  pending: "Pending / queued",
  cancelled: "Cancelled / skipped",
};

const STATUS_GROUP_ORDER: StatusGroup[] = [
  "running",
  "pending",
  "succeeded",
  "failed",
  "cancelled",
];

const PENDING_BAR_DURATION_MIN = 5;

const groupForStatus = (status: string): StatusGroup => {
  switch (status.toLowerCase()) {
    case "running":
      return "running";
    case "succeeded":
      return "succeeded";
    case "failed":
    case "timed_out":
    case "lost":
      return "failed";
    case "cancelled":
    case "skipped":
      return "cancelled";
    default:
      return "pending";
  }
};

const safeDate = (value: string | null | undefined): Date | null => {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const buildRunTask = (run: WorkspaceRunRow): GanttTask | null => {
  const execStarts = run.executions
    .map((exec) => safeDate(exec.startedAt))
    .filter((d): d is Date => d !== null);
  const execEnds = run.executions
    .map((exec) => safeDate(exec.finishedAt))
    .filter((d): d is Date => d !== null);

  const created = safeDate(run.createdAt);
  const start =
    execStarts.length > 0
      ? new Date(Math.min(...execStarts.map((d) => d.getTime())))
      : created;
  if (!start) return null;

  const isOpen = run.status.toLowerCase() === "running" || execStarts.length === 0;
  const end = isOpen
    ? new Date(Math.max(start.getTime() + PENDING_BAR_DURATION_MIN * 60_000, Date.now()))
    : execEnds.length > 0
      ? new Date(Math.max(...execEnds.map((d) => d.getTime())))
      : new Date(start.getTime() + PENDING_BAR_DURATION_MIN * 60_000);

  const hasNoStart = execStarts.length === 0;
  const hover =
    `<b>${run.name}</b><br>` +
    `${run.projectName} · ${run.experimentName}<br>` +
    `Status: ${run.status}<br>` +
    `Backend: ${run.backend ?? "—"}${run.cluster ? ` · ${run.cluster}` : ""}<br>` +
    `Executions: ${run.executionCount}<br>` +
    `Started: ${start.toLocaleString()}<br>` +
    (hasNoStart ? "(estimated — queued)" : `Ended: ${end.toLocaleString()}`);

  return {
    runId: run.id,
    executionId: null,
    label: run.name,
    start,
    end,
    statusGroup: groupForStatus(run.status),
    statusRaw: run.status,
    hover,
    isPending: hasNoStart,
  };
};

const buildExecutionTasks = (run: WorkspaceRunRow): GanttTask[] => {
  const tasks: GanttTask[] = [];
  for (const exec of run.executions) {
    const start = safeDate(exec.startedAt) ?? safeDate(run.createdAt);
    if (!start) continue;
    const isOpen = exec.status.toLowerCase() === "running" || !exec.finishedAt;
    const end = isOpen
      ? new Date(Math.max(start.getTime() + PENDING_BAR_DURATION_MIN * 60_000, Date.now()))
      : (safeDate(exec.finishedAt) ??
        new Date(start.getTime() + PENDING_BAR_DURATION_MIN * 60_000));

    const hover =
      `<b>${run.name}</b> · ${exec.executionId.slice(0, 12)}<br>` +
      `Status: ${exec.status}<br>` +
      `Backend: ${exec.backend ?? "—"}<br>` +
      (exec.schedulerJobId ? `Scheduler job: ${exec.schedulerJobId}<br>` : "") +
      `Started: ${start.toLocaleString()}<br>` +
      (isOpen ? "(in progress)" : `Ended: ${end.toLocaleString()}`);

    tasks.push({
      runId: run.id,
      executionId: exec.executionId,
      label: `${run.name} · ${exec.executionId.slice(0, 8)}`,
      start,
      end,
      statusGroup: groupForStatus(exec.status),
      statusRaw: exec.status,
      hover,
      isPending: !exec.startedAt,
    });
  }
  return tasks;
};

export const RunsGanttChart = ({
  rows,
  mode,
  onSelectRun,
  onSelectExecution,
}: RunsGanttChartProps): JSX.Element => {
  const tasks = useMemo<GanttTask[]>(() => {
    if (mode === "runs") {
      const out: GanttTask[] = [];
      for (const row of rows) {
        const task = buildRunTask(row);
        if (task) out.push(task);
      }
      return out;
    }
    return rows.flatMap(buildExecutionTasks);
  }, [rows, mode]);

  // Plotly's recommended Gantt pattern: one scatter trace per status group, with
  // each task represented as a thick line segment between (start, end) on the
  // shared categorical y-axis. Grouping by status gives us a free legend.
  // See: https://plotly.com/javascript/gantt/
  const traces = useMemo(() => {
    const yOrder = tasks.map((task) => task.label);
    const grouped = new Map<StatusGroup, GanttTask[]>();
    for (const task of tasks) {
      const list = grouped.get(task.statusGroup) ?? [];
      list.push(task);
      grouped.set(task.statusGroup, list);
    }

    return STATUS_GROUP_ORDER.filter((group) => grouped.has(group)).map((group) => {
      const groupTasks = grouped.get(group) ?? [];
      const x: Array<string | null> = [];
      const y: Array<string | null> = [];
      const customdata: Array<{ runId: string; executionId: string | null } | null> = [];
      const hovertemplate: string[] = [];

      for (const task of groupTasks) {
        x.push(task.start.toISOString(), task.end.toISOString(), null);
        y.push(task.label, task.label, null);
        const cd = { runId: task.runId, executionId: task.executionId };
        customdata.push(cd, cd, null);
        const hover = `${task.hover}<extra></extra>`;
        hovertemplate.push(hover, hover, "");
      }

      return {
        type: "scatter",
        mode: "lines",
        name: STATUS_GROUP_LABEL[group],
        x,
        y,
        customdata,
        hovertemplate,
        line: {
          color: STATUS_GROUP_COLOR[group],
          width: 18,
        },
        opacity: group === "pending" ? 0.55 : 1,
        connectgaps: false,
        // Keep y-axis ordering stable across traces.
        yaxis: "y",
        yref: "y",
        meta: yOrder,
      };
    });
  }, [tasks]);

  const handleClick = (event: { points: Array<{ customdata?: unknown }> }): void => {
    const point = event.points?.[0];
    if (!point) return;
    const data = point.customdata as { runId: string; executionId: string | null } | undefined;
    if (!data) return;
    const run = rows.find((r) => r.id === data.runId);
    if (!run) return;
    if (data.executionId) {
      const exec = run.executions.find((e) => e.executionId === data.executionId);
      if (exec) onSelectExecution(run, exec);
    } else {
      onSelectRun(run);
    }
  };

  const layout = useMemo(
    () => ({
      height: Math.max(220, Math.min(720, 28 * tasks.length + 90)),
      margin: { l: 240, r: 24, t: 12, b: 40 },
      xaxis: {
        type: "date" as const,
        showgrid: true,
        gridcolor: "rgba(125,125,125,0.15)",
      },
      yaxis: {
        type: "category" as const,
        categoryorder: "array" as const,
        categoryarray: tasks.map((task) => task.label).reverse(),
        automargin: true,
        tickfont: { size: 11 },
      },
      hovermode: "closest" as const,
      legend: { orientation: "h" as const, y: -0.18, x: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    }),
    [tasks],
  );

  if (tasks.length === 0) {
    return (
      <div className="flex h-full min-h-[280px] items-center justify-center rounded border border-dashed border-border bg-muted/20 px-6 py-12 text-center text-sm text-muted-foreground">
        No timeline data yet — runs without start times are hidden from the chart.
      </div>
    );
  }

  return (
    <div className="rounded border border-border bg-background">
      <Plot
        data={traces}
        layout={layout}
        config={GANTT_CONFIG}
        style={GANTT_STYLE}
        useResizeHandler
        onClick={handleClick}
      />
    </div>
  );
};

const GANTT_CONFIG = { displayModeBar: false, responsive: true };
const GANTT_STYLE = { width: "100%" };
