import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight, Table2 } from "lucide-react";
import type { JSX, ReactNode } from "react";
import { useEffect, useMemo } from "react";

import { EmptyState, StatusBadge } from "@/app/components/entity";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatDuration, formatRelative } from "@/lib/format-time";
import { cn } from "@/lib/utils";

import {
  computeRunDurationSeconds,
  type JobsSort,
  type JobsSortKey,
  nextJobsSort,
  PAGE_SIZE_OPTIONS,
  paginate,
  sortJobs,
} from "./jobsTable";
import type { WorkspaceRunRow } from "./types";

interface RunsJobsTableProps {
  rows: WorkspaceRunRow[];
  selectedRunId: string | null;
  onSelectRun: (run: WorkspaceRunRow) => void;
  sort: JobsSort;
  onSortChange: (next: JobsSort) => void;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
}

interface ColumnDef {
  key: JobsSortKey;
  label: string;
  align?: "left" | "right";
  className?: string;
}

const COLUMNS: ColumnDef[] = [
  { key: "status", label: "Status", className: "w-[120px]" },
  { key: "name", label: "Run" },
  { key: "project", label: "Project · Experiment" },
  { key: "backend", label: "Backend" },
  { key: "attempts", label: "Attempts", align: "right" },
  { key: "duration", label: "Duration", align: "right" },
  { key: "submitted", label: "Submitted", align: "right" },
];

/**
 * Workspace Jobs table — sortable columns + client-side pagination.
 * Sort / page state is owned by the parent (URL-backed).
 */
export const RunsJobsTable = ({
  rows,
  selectedRunId,
  onSelectRun,
  sort,
  onSortChange,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: RunsJobsTableProps): JSX.Element => {
  const sorted = useMemo(() => sortJobs(rows, sort), [rows, sort]);
  const slice = useMemo(() => paginate(sorted, page, pageSize), [sorted, page, pageSize]);

  // Parent may still hold a page past the end after a filter shrink.
  useEffect(() => {
    if (slice.page !== page) onPageChange(slice.page);
  }, [slice.page, page, onPageChange]);

  if (rows.length === 0) {
    return (
      <div className="flex h-full min-h-[240px] items-center justify-center rounded-lg border border-dashed border-border bg-card">
        <EmptyState
          icon={<Table2 className="h-5 w-5" />}
          title="No matching runs"
          description="Adjust filters in the sidebar, or clear them to see the full workspace."
        />
      </div>
    );
  }

  const rangeStart = slice.totalItems === 0 ? 0 : (slice.page - 1) * slice.pageSize + 1;
  const rangeEnd = Math.min(slice.page * slice.pageSize, slice.totalItems);

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-hidden rounded-lg border border-border bg-card shadow-none">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-border/60 bg-muted/20">
              <tr className="text-xs text-muted-foreground">
                {COLUMNS.map((col) => (
                  <SortableTh
                    key={col.key}
                    column={col}
                    active={sort.key === col.key}
                    dir={sort.dir}
                    onClick={() => onSortChange(nextJobsSort(sort, col.key))}
                  />
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              {slice.items.map((run) => {
                const isSelected = run.id === selectedRunId;
                const duration = computeRunDurationSeconds(run);
                return (
                  <tr
                    key={run.id}
                    onClick={() => onSelectRun(run)}
                    className={cn(
                      "cursor-pointer transition-colors",
                      isSelected ? "bg-primary/5" : "hover:bg-muted/40",
                    )}
                  >
                    <Td className="align-middle">
                      <StatusBadge status={run.status} size="sm" dot />
                    </Td>
                    <Td className="align-middle">
                      <div className="min-w-0">
                        <p className="truncate font-medium text-foreground">{run.name || run.id}</p>
                        <p
                          className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground"
                          title={run.id}
                        >
                          {run.id}
                        </p>
                      </div>
                    </Td>
                    <Td className="align-middle text-muted-foreground">
                      <div className="min-w-0">
                        <p className="truncate text-foreground">{run.projectName}</p>
                        <p className="truncate text-xs">{run.experimentName}</p>
                      </div>
                    </Td>
                    <Td className="align-middle text-muted-foreground">
                      {run.backend ? (
                        <div className="min-w-0">
                          <p className="truncate text-foreground">{run.backend}</p>
                          {run.cluster && (
                            <p className="truncate font-mono text-[11px]">{run.cluster}</p>
                          )}
                        </div>
                      ) : (
                        <span>—</span>
                      )}
                    </Td>
                    <Td className="text-right align-middle tabular-nums text-muted-foreground">
                      {run.executionCount}
                    </Td>
                    <Td className="text-right align-middle font-mono text-xs tabular-nums text-muted-foreground">
                      {formatDuration(duration)}
                    </Td>
                    <Td className="text-right align-middle text-xs text-muted-foreground">
                      {formatRelative(run.createdAt)}
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <div className="tabular-nums">
          {rangeStart}–{rangeEnd} of {slice.totalItems}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1.5">
            <span>Rows</span>
            <Select
              value={String(pageSize)}
              onValueChange={(value) => onPageSizeChange(Number.parseInt(value, 10))}
            >
              <SelectTrigger size="sm" className="h-7 w-[72px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PAGE_SIZE_OPTIONS.map((size) => (
                  <SelectItem key={size} value={String(size)}>
                    {size}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="h-7 w-7"
              disabled={slice.page <= 1}
              onClick={() => onPageChange(slice.page - 1)}
              aria-label="Previous page"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <span className="min-w-[4.5rem] text-center tabular-nums">
              {slice.page} / {slice.totalPages}
            </span>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="h-7 w-7"
              disabled={slice.page >= slice.totalPages}
              onClick={() => onPageChange(slice.page + 1)}
              aria-label="Next page"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

const SortableTh = ({
  column,
  active,
  dir,
  onClick,
}: {
  column: ColumnDef;
  active: boolean;
  dir: JobsSort["dir"];
  onClick: () => void;
}): JSX.Element => {
  const Icon = !active ? ArrowUpDown : dir === "asc" ? ArrowUp : ArrowDown;
  return (
    <th
      className={cn(
        "px-3 py-2.5 font-medium",
        column.align === "right" ? "text-right" : "text-left",
        column.className,
      )}
    >
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "inline-flex items-center gap-1 transition-colors hover:text-foreground",
          column.align === "right" && "flex-row-reverse",
          active ? "text-foreground" : "text-muted-foreground",
        )}
      >
        {column.label}
        <Icon className="h-3 w-3 opacity-70" aria-hidden="true" />
        <span className="sr-only">
          {active ? `sorted ${dir}` : "sort"}
        </span>
      </button>
    </th>
  );
};

const Td = ({ children, className }: { children: ReactNode; className?: string }): JSX.Element => (
  <td className={cn("px-3 py-2.5", className)}>{children}</td>
);
