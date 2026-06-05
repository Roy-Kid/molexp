import { ExternalLink, GitCompareArrows } from "lucide-react";
import type { JSX } from "react";
import { useMemo } from "react";
import { EmptyState, StatusBadge } from "@/app/components/entity";
import { formatScalar } from "@/app/renderers/dashboardData";
import type { RunSummary } from "@/app/types";
import { cn } from "@/lib/utils";

interface ExperimentCompareProps {
  runs: RunSummary[];
  onOpenRun: (runId: string) => void;
}

interface CompareRow {
  key: string;
  /** Per-run cell values, aligned with the runs column order. */
  values: string[];
  /** True when not every run shares the same value — the interesting rows. */
  varies: boolean;
}

const buildRows = (
  runs: RunSummary[],
  pick: (run: RunSummary) => Record<string, unknown>,
): CompareRow[] => {
  const keys: string[] = [];
  const seen = new Set<string>();
  for (const run of runs) {
    for (const key of Object.keys(pick(run) ?? {})) {
      if (!seen.has(key)) {
        seen.add(key);
        keys.push(key);
      }
    }
  }
  return keys.map((key) => {
    const values = runs.map((run) => {
      const raw = pick(run)?.[key];
      return raw === undefined ? "—" : formatScalar(raw);
    });
    const varies = new Set(values).size > 1;
    return { key, values, varies };
  });
};

const RowGroup = ({
  title,
  rows,
  runCount,
}: {
  title: string;
  rows: CompareRow[];
  runCount: number;
}): JSX.Element | null => {
  if (rows.length === 0) return null;
  const variedCount = rows.filter((r) => r.varies).length;
  return (
    <>
      <tr>
        <td
          colSpan={runCount + 1}
          className="border-b border-border/60 bg-muted/40 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground"
        >
          {title}
          <span className="ml-2 font-normal normal-case text-muted-foreground/70">
            {variedCount > 0 ? `${variedCount} differ` : "all identical"}
          </span>
        </td>
      </tr>
      {rows.map((row) => (
        <tr
          key={`${title}:${row.key}`}
          className={cn(
            "border-b border-border/40 last:border-b-0",
            row.varies ? "bg-amber-500/[0.04]" : "",
          )}
        >
          <th
            scope="row"
            className={cn(
              "sticky left-0 z-10 max-w-[180px] truncate border-r border-border/60 bg-background px-3 py-1.5 text-left align-top font-mono text-xs font-medium",
              row.varies
                ? "border-l-2 border-l-amber-500 text-foreground"
                : "text-muted-foreground",
            )}
            title={row.key}
          >
            {row.key}
          </th>
          {row.values.map((value, idx) => (
            <td
              key={`${row.key}:${runs[idx]?.id ?? idx}`}
              className={cn(
                "border-r border-border/40 px-3 py-1.5 align-top font-mono text-xs last:border-r-0",
                row.varies ? "text-foreground" : "text-muted-foreground",
                value === "—" && "text-muted-foreground/50",
              )}
            >
              <span className="block max-w-[200px] truncate" title={value}>
                {value}
              </span>
            </td>
          ))}
        </tr>
      ))}
    </>
  );
};

/**
 * Compares every run in an experiment field-by-field: a matrix of parameters
 * and results across runs, with the rows that actually differ pulled to the
 * eye. Real data straight from the run summaries — no mocks, no run pickers.
 */
export const ExperimentCompare = ({ runs }: ExperimentCompareProps): JSX.Element => {
  const ordered = useMemo(
    () =>
      [...runs].sort((a, b) => {
        const aT = Date.parse(a.startedAt ?? a.updatedAt ?? "") || 0;
        const bT = Date.parse(b.startedAt ?? b.updatedAt ?? "") || 0;
        return aT - bT;
      }),
    [runs],
  );

  const paramRows = useMemo(() => buildRows(ordered, (r) => r.parameters), [ordered]);
  const resultRows = useMemo(() => buildRows(ordered, (r) => r.results), [ordered]);

  if (runs.length < 2) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState
          icon={<GitCompareArrows className="h-6 w-6" />}
          title="Need at least two runs to compare"
          description="Launch more runs in this experiment to see how their parameters and results line up."
        />
      </div>
    );
  }

  if (paramRows.length === 0 && resultRows.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState
          icon={<GitCompareArrows className="h-6 w-6" />}
          title="Nothing to compare yet"
          description="These runs have no recorded parameters or results."
        />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-auto">
      <table className="w-full border-collapse text-xs">
        <thead className="sticky top-0 z-20">
          <tr>
            <th className="sticky left-0 z-30 border-b border-r border-border/60 bg-muted/60 px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Field
            </th>
            {ordered.map((run) => (
              <th
                key={run.id}
                className="min-w-[140px] border-b border-r border-border/60 bg-muted/60 px-3 py-2 text-left align-bottom last:border-r-0"
              >
                <button
                  type="button"
                  onClick={() => onOpenRun(run.id)}
                  className="group flex items-center gap-1 font-mono text-xs font-medium text-foreground hover:text-primary"
                  title={`Open ${run.name || run.id}`}
                >
                  <span className="max-w-[120px] truncate">
                    {run.name || run.id.substring(0, 8)}
                  </span>
                  <ExternalLink className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" />
                </button>
                <div className="mt-1 flex items-center gap-1.5">
                  <StatusBadge status={run.status} size="sm" />
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {run.id.substring(0, 8)}
                  </span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <RowGroup title="Parameters" rows={paramRows} runCount={ordered.length} />
          <RowGroup title="Results" rows={resultRows} runCount={ordered.length} />
        </tbody>
      </table>
      <div className="flex items-center gap-2 border-t border-border/60 px-3 py-2 text-[11px] text-muted-foreground">
        <span className="inline-block h-2.5 w-1 rounded-sm bg-amber-500" />
        <span>Highlighted rows differ across runs.</span>
      </div>
    </div>
  );
};
