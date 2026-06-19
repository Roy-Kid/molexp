import type { JSX } from "react";
import { useEffect, useMemo, useState } from "react";

import { MolplotGanttChart } from "@/plugins/molplot";

import type { WorkspaceExecutionRow, WorkspaceRunRow } from "./types";

/** Tick the wall-clock-derived end time for active bars at this cadence. */
const LIVE_TICK_MS = 5_000;

interface RunsGanttChartProps {
  rows: WorkspaceRunRow[];
  mode: "runs" | "executions";
  onSelectRun: (run: WorkspaceRunRow) => void;
  onSelectExecution: (run: WorkspaceRunRow, execution: WorkspaceExecutionRow) => void;
}

interface GanttTaskLocal {
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

const buildRunTask = (run: WorkspaceRunRow, nowMs: number): GanttTaskLocal | null => {
  const execStarts = run.executions
    .map((exec) => safeDate(exec.startedAt))
    .filter((d): d is Date => d !== null);
  const execEnds = run.executions
    .map((exec) => safeDate(exec.finishedAt))
    .filter((d): d is Date => d !== null);

  const created = safeDate(run.createdAt);
  const start =
    execStarts.length > 0 ? new Date(Math.min(...execStarts.map((d) => d.getTime()))) : created;
  if (!start) return null;

  const isOpen = run.status.toLowerCase() === "running" || execStarts.length === 0;
  const end = isOpen
    ? new Date(Math.max(start.getTime() + PENDING_BAR_DURATION_MIN * 60_000, nowMs))
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

const buildExecutionTasks = (run: WorkspaceRunRow, nowMs: number): GanttTaskLocal[] => {
  const tasks: GanttTaskLocal[] = [];
  for (const exec of run.executions) {
    const start = safeDate(exec.startedAt) ?? safeDate(run.createdAt);
    if (!start) continue;
    const isOpen = exec.status.toLowerCase() === "running" || !exec.finishedAt;
    const end = isOpen
      ? new Date(Math.max(start.getTime() + PENDING_BAR_DURATION_MIN * 60_000, nowMs))
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
  // Tick `nowMs` periodically so in-progress bars keep growing without
  // waiting for `rows` to refetch — `buildRunTask`/`buildExecutionTasks`
  // pin open-ended `end` to this value rather than `Date.now()`.
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setNowMs(Date.now());
    }, LIVE_TICK_MS);
    return () => window.clearInterval(intervalId);
  }, []);

  const tasks = useMemo<GanttTaskLocal[]>(() => {
    if (mode === "runs") {
      const out: GanttTaskLocal[] = [];
      for (const row of rows) {
        const task = buildRunTask(row, nowMs);
        if (task) out.push(task);
      }
      return out;
    }
    return rows.flatMap((row) => buildExecutionTasks(row, nowMs));
  }, [rows, mode, nowMs]);

  const config = useMemo(
    () => ({
      tasks: tasks.map((task) => ({
        id: `${task.runId}::${task.executionId ?? ""}`,
        label: task.label,
        start: task.start,
        end: task.end,
        statusGroup: task.statusGroup,
        hover: task.hover,
        customdata: { runId: task.runId, executionId: task.executionId },
      })),
      statusColors: STATUS_GROUP_COLOR,
      statusLabels: STATUS_GROUP_LABEL,
      statusOrder: STATUS_GROUP_ORDER,
      statusOpacity: { pending: 0.55 },
      showLegend: true,
      theme: "auto" as const,
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
      <MolplotGanttChart
        config={config}
        onTaskClick={(event) => {
          const data = event.customdata as
            | { runId: string; executionId: string | null }
            | undefined;
          if (!data) return;
          const run = rows.find((r) => r.id === data.runId);
          if (!run) return;
          if (data.executionId) {
            const exec = run.executions.find((e) => e.executionId === data.executionId);
            if (exec) onSelectExecution(run, exec);
          } else {
            onSelectRun(run);
          }
        }}
        style={{ width: "100%" }}
      />
    </div>
  );
};
