import { Table2 } from "lucide-react";
import type { JSX } from "react";

import { StatusBadge } from "@/app/components/entity";
import { formatDuration, formatRelative } from "@/lib/format-time";

import type { WorkspaceRunRow } from "./types";

interface RunsJobsTableProps {
  rows: WorkspaceRunRow[];
  selectedRunId: string | null;
  onSelectRun: (run: WorkspaceRunRow) => void;
}

const computeRunDurationSeconds = (run: WorkspaceRunRow): number | null => {
  const start = run.executions
    .map((e) => (e.startedAt ? new Date(e.startedAt).getTime() : NaN))
    .filter((v) => !Number.isNaN(v))
    .sort((a, b) => a - b)[0];
  if (typeof start !== "number") return null;
  const end = run.finishedAt ? new Date(run.finishedAt).getTime() : Date.now();
  if (Number.isNaN(end)) return null;
  return Math.max(0, (end - start) / 1000);
};

/**
 * Compact runs table — minimum viable Jobs tab body. PR2 will add column
 * sorting, pagination, per-row "..." menus and URL-persisted state.
 */
export const RunsJobsTable = ({
  rows,
  selectedRunId,
  onSelectRun,
}: RunsJobsTableProps): JSX.Element => {
  if (rows.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-card p-10 text-center text-xs text-muted-foreground">
        <Table2 className="h-5 w-5 opacity-40" />
        <p>No runs match the current filters.</p>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-border/60 bg-card">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="border-b border-border bg-muted/30">
            <tr className="text-[10px] uppercase tracking-wide text-muted-foreground">
              <Th className="w-[120px]">Status</Th>
              <Th>Run</Th>
              <Th>Project · Experiment</Th>
              <Th>Backend</Th>
              <Th className="text-right">Attempts</Th>
              <Th className="text-right">Duration</Th>
              <Th className="text-right">Submitted</Th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60">
            {rows.map((run) => {
              const isSelected = run.id === selectedRunId;
              const duration = computeRunDurationSeconds(run);
              return (
                <tr
                  key={run.id}
                  onClick={() => onSelectRun(run)}
                  className={
                    isSelected
                      ? "cursor-pointer bg-info-soft/60"
                      : "cursor-pointer hover:bg-accent/40"
                  }
                >
                  <Td className="align-top">
                    <StatusBadge status={run.status} size="sm" dot />
                  </Td>
                  <Td className="align-top">
                    <div className="min-w-0">
                      <p className="truncate font-medium text-foreground">{run.name || run.id}</p>
                      <p
                        className="mt-0.5 truncate font-mono text-[10px] text-muted-foreground"
                        title={run.id}
                      >
                        {run.id}
                      </p>
                    </div>
                  </Td>
                  <Td className="align-top text-muted-foreground">
                    <div className="min-w-0">
                      <p className="truncate text-foreground">{run.projectName}</p>
                      <p className="truncate">{run.experimentName}</p>
                    </div>
                  </Td>
                  <Td className="align-top text-muted-foreground">
                    {run.backend ? (
                      <div className="min-w-0">
                        <p className="truncate text-foreground">{run.backend}</p>
                        {run.cluster && <p className="truncate">{run.cluster}</p>}
                      </div>
                    ) : (
                      <span>—</span>
                    )}
                  </Td>
                  <Td className="text-right tabular-nums text-muted-foreground">
                    {run.executionCount}
                  </Td>
                  <Td className="text-right tabular-nums text-muted-foreground">
                    {formatDuration(duration)}
                  </Td>
                  <Td className="text-right text-muted-foreground">
                    {formatRelative(run.createdAt)}
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const Th = ({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}): JSX.Element => (
  <th className={`px-3 py-2 text-left font-semibold ${className ?? ""}`}>{children}</th>
);

const Td = ({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}): JSX.Element => <td className={`px-3 py-2 ${className ?? ""}`}>{children}</td>;
