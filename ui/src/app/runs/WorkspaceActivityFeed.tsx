import type { JSX } from "react";
import { useEffect, useState } from "react";

import { WorkspaceService } from "@/api/generated";
import { formatRelative, formatTimestamp } from "@/lib/format-time";
import { cn } from "@/lib/utils";

import {
  eventVisualFor,
  FEED_EMPTY_TEXT,
  resolveEventRef,
  type WorkspaceEventRow,
} from "./activityFeed";
import { POLL_INTERVAL_MS } from "./useWorkspaceRuns";

interface WorkspaceActivityFeedProps {
  /** Run ids the snapshot currently knows — resolvable refs become links. */
  knownRunIds: ReadonlySet<string>;
  onSelectRun: (runId: string) => void;
  onOpenKnowledge: (path: string) => void;
  max?: number;
}

/**
 * The workspace-wide "what just happened" feed (vision-loop-12): a poll over
 * `GET /api/events` — the event spine's global read — rendered with entity
 * links resolved against the snapshot.
 */
export const WorkspaceActivityFeed = ({
  knownRunIds,
  onSelectRun,
  onOpenKnowledge,
  max = 20,
}: WorkspaceActivityFeedProps): JSX.Element => {
  const [events, setEvents] = useState<WorkspaceEventRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async (): Promise<void> => {
      try {
        const rows = await WorkspaceService.getWorkspaceEventsApiEventsGet(
          undefined,
          undefined,
          max,
        );
        if (!cancelled) {
          setEvents(rows as WorkspaceEventRow[]);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(String(err));
      }
    };
    void tick();
    const id = setInterval(() => void tick(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [max]);

  if (error && events.length === 0) {
    return <p className="text-sm text-muted-foreground">{error}</p>;
  }
  if (events.length === 0) {
    return <p className="text-sm text-muted-foreground">{FEED_EMPTY_TEXT}</p>;
  }

  return (
    <ol className="space-y-0.5">
      {events.map((event) => {
        const visual = eventVisualFor(event.type);
        const Icon = visual.icon;
        return (
          <li
            key={event.id}
            className="flex items-start gap-2.5 rounded-md px-1.5 py-1.5 text-xs transition-colors hover:bg-muted/40"
          >
            <span
              aria-hidden="true"
              className={cn(
                "mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full",
                visual.dotClass,
              )}
            >
              <Icon className="h-2.5 w-2.5 text-background" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate font-medium text-foreground">
                  {visual.label}
                  <span className="ml-1.5 font-normal text-muted-foreground">· {event.actor}</span>
                </span>
                <span
                  className="shrink-0 tabular-nums text-[11px] text-muted-foreground"
                  title={formatTimestamp(event.created_at)}
                >
                  {formatRelative(event.created_at)}
                </span>
              </div>
              <span className="mt-0.5 flex flex-wrap gap-x-2 font-mono text-[11px] text-muted-foreground">
                {event.refs.map((ref) => {
                  const resolved = resolveEventRef(ref, event.type, knownRunIds, event.payload);
                  if (resolved.kind === "run") {
                    return (
                      <button
                        key={ref}
                        type="button"
                        className="truncate text-primary underline-offset-2 hover:underline"
                        onClick={() => onSelectRun(resolved.runId)}
                      >
                        {resolved.text}
                      </button>
                    );
                  }
                  if (resolved.kind === "knowledge") {
                    return (
                      <button
                        key={ref}
                        type="button"
                        className="truncate text-primary underline-offset-2 hover:underline"
                        onClick={() => onOpenKnowledge(resolved.path)}
                      >
                        {resolved.text}
                      </button>
                    );
                  }
                  return (
                    <span key={ref} className="truncate">
                      {resolved.text}
                    </span>
                  );
                })}
              </span>
            </div>
          </li>
        );
      })}
    </ol>
  );
};
