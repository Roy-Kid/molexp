import { CheckCircle2, CircleDot, FlagTriangleRight, Send, XCircle } from "lucide-react";
import type { JSX } from "react";
import { formatRelative, formatTimestamp } from "@/lib/format-time";
import { cn } from "@/lib/utils";

import { computeRecentEventsForRun, type RecentEvent, type RecentEventKind } from "./aggregates";
import { groupForStatus } from "./statusGroups";
import type { WorkspaceRunRow } from "./types";

interface RunsRecentEventsProps {
  run: WorkspaceRunRow;
  max?: number;
}

const ICONS: Record<RecentEventKind, typeof Send> = {
  submitted: Send,
  started: CircleDot,
  finished: FlagTriangleRight,
};

const finishedIconFor = (outcome: string | undefined): typeof Send => {
  const group = groupForStatus(outcome ?? null);
  if (group === "succeeded") return CheckCircle2;
  if (group === "failed") return XCircle;
  return FlagTriangleRight;
};

const dotClassFor = (event: RecentEvent): string => {
  if (event.kind === "submitted") return "bg-muted-foreground/40";
  if (event.kind === "started") return "bg-info";
  const group = groupForStatus(event.outcome ?? null);
  if (group === "succeeded") return "bg-success";
  if (group === "failed") return "bg-destructive";
  if (group === "cancelled") return "bg-muted-foreground/60";
  return "bg-muted-foreground/40";
};

const labelFor = (event: RecentEvent): string => {
  switch (event.kind) {
    case "submitted":
      return "Submitted";
    case "started":
      return "Execution started";
    case "finished": {
      const group = groupForStatus(event.outcome ?? null);
      if (group === "succeeded") return "Execution succeeded";
      if (group === "failed") return "Execution failed";
      if (group === "cancelled") return "Execution cancelled";
      return `Execution finished (${event.outcome ?? "unknown"})`;
    }
  }
};

export const RunsRecentEvents = ({ run, max = 8 }: RunsRecentEventsProps): JSX.Element => {
  const events = computeRecentEventsForRun(run).slice(0, max);

  if (events.length === 0) {
    return (
      <p className="text-xs italic text-muted-foreground">
        No timeline events yet — backend has not surfaced submitted/started/finished timestamps for
        this run.
      </p>
    );
  }

  return (
    <ol className="space-y-2">
      {events.map((event) => {
        const Icon = event.kind === "finished" ? finishedIconFor(event.outcome) : ICONS[event.kind];
        return (
          <li
            key={`${event.kind}-${event.executionId ?? "run"}-${event.at}`}
            className="flex items-start gap-2 text-xs"
          >
            <span
              aria-hidden="true"
              className={cn(
                "mt-1 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full",
                dotClassFor(event),
              )}
            >
              <Icon className="h-2.5 w-2.5 text-background" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate font-medium text-foreground">{labelFor(event)}</span>
                <span
                  className="shrink-0 text-[10px] tabular-nums text-muted-foreground"
                  title={formatTimestamp(event.at)}
                >
                  {formatRelative(event.at)}
                </span>
              </div>
              {event.executionId && (
                <span className="block truncate font-mono text-[10px] text-muted-foreground">
                  {event.executionId}
                </span>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
};
