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
    return <p className="text-sm text-muted-foreground">No runs match the current filters.</p>;
  }

  const visible = segments.filter((segment) => segment.count > 0);

  return (
    <div className="space-y-3">
      <div
        role="img"
        aria-label={`Status distribution across ${total} runs`}
        className="flex h-2 w-full overflow-hidden rounded-full bg-muted"
      >
        {visible.map((segment) => {
          const widthPct = segment.ratio * 100;
          return (
            <button
              key={segment.spec.id}
              type="button"
              onClick={onSelectStatus ? () => onSelectStatus(segment.spec.filterValue) : undefined}
              title={`${segment.spec.label}: ${segment.count} (${(segment.ratio * 100).toFixed(1)}%)`}
              className={cn(
                "h-full min-w-[2px] transition-opacity hover:opacity-80",
                onSelectStatus ? "cursor-pointer" : "cursor-default",
              )}
              style={{ width: `${widthPct}%`, backgroundColor: segment.spec.color }}
              aria-label={`${segment.spec.label}: ${segment.count} runs`}
            />
          );
        })}
      </div>
      <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
        {segments.map((segment) => {
          const dimmed = segment.count === 0;
          const clickable = onSelectStatus !== undefined && !dimmed;
          return (
            <li key={segment.spec.id}>
              <button
                type="button"
                onClick={clickable ? () => onSelectStatus(segment.spec.filterValue) : undefined}
                disabled={!clickable}
                className={cn(
                  "flex w-full items-center justify-between gap-2 text-xs transition-colors",
                  clickable
                    ? "cursor-pointer text-foreground hover:text-primary"
                    : "cursor-default",
                  dimmed && "opacity-40",
                )}
              >
                <span className="inline-flex min-w-0 items-center gap-2 text-muted-foreground">
                  <span
                    aria-hidden="true"
                    className="h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ backgroundColor: segment.spec.color }}
                  />
                  <span className="truncate">{segment.spec.label}</span>
                </span>
                <span className="flex shrink-0 items-center gap-1.5 tabular-nums">
                  <span className="font-medium text-foreground">{segment.count}</span>
                  <span className="w-8 text-right text-muted-foreground">
                    {(segment.ratio * 100).toFixed(0)}%
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
};
