import { ExternalLink, GitCompareArrows } from "lucide-react";
import type { JSX } from "react";
import { useMemo } from "react";
import { EmptyState } from "@/app/components/entity";
import { formatScalar } from "@/app/renderers/dashboardData";
import type { RunSummary } from "@/app/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { RunStatusBadge, WorkbenchAction } from "@/components/workbench";
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
  runIds,
}: {
  title: string;
  rows: CompareRow[];
  /** Ordered run ids, aligned with each row's `values` — used for stable keys. */
  runIds: string[];
}): JSX.Element | null => {
  if (rows.length === 0) return null;
  const variedCount = rows.filter((r) => r.varies).length;
  return (
    <>
      <TableRow>
        <TableCell
          colSpan={runIds.length + 1}
          className="border-b border-border/60 bg-muted/40 px-3 py-2 text-micro font-semibold uppercase tracking-wide text-muted-foreground"
        >
          {title}
          <span className="ml-2 font-normal normal-case text-muted-foreground/70">
            {variedCount > 0 ? `${variedCount} differ` : "all identical"}
          </span>
        </TableCell>
      </TableRow>
      {rows.map((row) => (
        <TableRow
          key={`${title}:${row.key}`}
          className={cn(
            "border-b border-border/40 last:border-b-0",
            row.varies ? "bg-diff-modified-soft" : "",
          )}
        >
          <TableHead
            scope="row"
            className={cn(
              "sticky left-0 z-10 max-w-44 truncate border-r border-border/60 bg-background px-3 py-2 text-left align-top font-mono text-label font-medium",
              row.varies
                ? "border-l border-l-diff-modified text-foreground"
                : "text-muted-foreground",
            )}
            title={row.key}
          >
            {row.key}
          </TableHead>
          {row.values.map((value, idx) => (
            <TableCell
              key={`${row.key}:${runIds[idx] ?? idx}`}
              className={cn(
                "border-r border-border/40 px-3 py-2 align-top font-mono text-label last:border-r-0",
                row.varies ? "text-foreground" : "text-muted-foreground",
                value === "—" && "text-muted-foreground/50",
              )}
            >
              <span className="block max-w-52 truncate" title={value}>
                {value}
              </span>
            </TableCell>
          ))}
        </TableRow>
      ))}
    </>
  );
};

/**
 * Compares every run in an experiment field-by-field: a matrix of parameters
 * and results across runs, with the rows that actually differ pulled to the
 * eye. Real data straight from the run summaries — no mocks, no run pickers.
 */
export const ExperimentCompare = ({ runs, onOpenRun }: ExperimentCompareProps): JSX.Element => {
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
      <Table className="w-full border-collapse text-label">
        <TableHeader className="sticky top-0 z-20">
          <TableRow>
            <TableHead className="sticky left-0 z-30 border-b border-r border-border/60 bg-muted/60 px-3 py-2 text-left text-micro font-semibold uppercase tracking-wide text-muted-foreground">
              Field
            </TableHead>
            {ordered.map((run) => (
              <TableHead
                key={run.id}
                className="min-w-36 border-b border-r border-border/60 bg-muted/60 px-3 py-2 text-left align-bottom last:border-r-0"
              >
                <WorkbenchAction
                  kind="ghost"
                  size="content"
                  type="button"
                  onClick={() => onOpenRun(run.id)}
                  className="group flex items-center gap-1 font-mono text-label font-medium text-foreground hover:text-accent"
                  title={`Open ${run.name || run.id}`}
                >
                  <span className="max-w-32 truncate">{run.name || run.id.substring(0, 8)}</span>
                  <ExternalLink className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" />
                </WorkbenchAction>
                <div className="mt-1 flex items-center gap-2">
                  <RunStatusBadge status={run.status} size="sm" />
                  <span className="font-mono text-micro text-muted-foreground">
                    {run.id.substring(0, 8)}
                  </span>
                </div>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          <RowGroup title="Parameters" rows={paramRows} runIds={ordered.map((r) => r.id)} />
          <RowGroup title="Results" rows={resultRows} runIds={ordered.map((r) => r.id)} />
        </TableBody>
      </Table>
      <div className="flex items-center gap-2 border-t border-border/60 px-3 py-2 text-micro text-muted-foreground">
        <span className="inline-block h-3 w-1 rounded-control bg-diff-modified" />
        <span>Highlighted rows differ across runs.</span>
      </div>
    </div>
  );
};
