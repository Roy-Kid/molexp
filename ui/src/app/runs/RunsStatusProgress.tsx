import type { JSX } from "react";
import { useMemo } from "react";

import { cn } from "@/lib/utils";

import { groupForStatus, STATUS_GROUPS, type StatusGroupSpec } from "./statusGroups";
import type { WorkspaceRunRow } from "./types";

interface RunsStatusProgressProps {
  runs: WorkspaceRunRow[];
  onSelectStatus?: (status: string) => void;
}

interface SegmentData {
  spec: StatusGroupSpec;
  count: number;
  ratio: number;
}

export const RunsStatusProgress = ({
  runs,
  onSelectStatus,
}: RunsStatusProgressProps): JSX.Element => {
  const { segments, total } = useMemo(() => {
    const counts = new Map<string, number>(STATUS_GROUPS.map((g) => [g.id, 0]));
    for (const run of runs) {
      const group = groupForStatus(run.status);
      if (group) counts.set(group, (counts.get(group) ?? 0) + 1);
    }
    const built: SegmentData[] = STATUS_GROUPS.map((spec) => {
      const count = counts.get(spec.id) ?? 0;
      return { spec, count, ratio: runs.length > 0 ? count / runs.length : 0 };
    });
    return { segments: built, total: runs.length };
  }, [runs]);

  if (total === 0) {
    return (
      <div className="text-center text-xs italic text-muted-foreground">
        No runs match the current filters.
      </div>
    );
  }

  const visible = segments.filter((segment) => segment.count > 0);

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          Status mix
        </div>
        <div className="text-[10px] tabular-nums text-muted-foreground">
          {total} run{total === 1 ? "" : "s"}
        </div>
      </div>
      <div
        role="img"
        aria-label={`Status distribution across ${total} runs`}
        className="flex h-3 w-full overflow-hidden rounded-full bg-muted"
      >
        {visible.map((segment, index) => {
          const widthPct = segment.ratio * 100;
          const isFirst = index === 0;
          const isLast = index === visible.length - 1;
          return (
            <button
              key={segment.spec.id}
              type="button"
              onClick={onSelectStatus ? () => onSelectStatus(segment.spec.filterValue) : undefined}
              title={`${segment.spec.label}: ${segment.count} (${(segment.ratio * 100).toFixed(1)}%)`}
              className={cn(
                "h-full transition-opacity hover:opacity-80",
                onSelectStatus ? "cursor-pointer" : "cursor-default",
                isFirst && "rounded-l-full",
                isLast && "rounded-r-full",
              )}
              style={{ width: `${widthPct}%`, backgroundColor: segment.spec.color }}
              aria-label={`${segment.spec.label}: ${segment.count} runs`}
            />
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px]">
        {segments.map((segment) => {
          const dimmed = segment.count === 0;
          const clickable = onSelectStatus !== undefined && !dimmed;
          return (
            <button
              key={segment.spec.id}
              type="button"
              onClick={clickable ? () => onSelectStatus(segment.spec.filterValue) : undefined}
              disabled={!clickable}
              className={cn(
                "flex items-center gap-1.5 transition-colors",
                clickable ? "cursor-pointer text-foreground hover:text-primary" : "cursor-default",
                dimmed && "opacity-40",
              )}
            >
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: segment.spec.color }}
              />
              <span className="text-muted-foreground">{segment.spec.label}</span>
              <span className="font-semibold tabular-nums">{segment.count}</span>
              <span className="text-muted-foreground">· {(segment.ratio * 100).toFixed(0)}%</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
