import type { JSX } from "react";
import { useEffect, useState } from "react";

import { WorkspaceService } from "@/api/generated";
import {
  WorkbenchAction,
  WorkbenchOperationState,
  WorkbenchRetryAction,
} from "@/components/workbench";
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
  /** Optional spine type filter (e.g. ``run.failed``); ``null``/undefined = all. */
  eventType?: string | null;
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
  eventType = null,
}: WorkspaceActivityFeedProps): JSX.Element => {
  const [events, setEvents] = useState<WorkspaceEventRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    void tick;
    let cancelled = false;
    const tickOnce = async (): Promise<void> => {
      try {
        const rows = await WorkspaceService.getWorkspaceEventsApiEventsGet(
          (eventType as
            | "run.created"
            | "run.started"
            | "run.failed"
            | "run.completed"
            | "asset.added"
            | "knowledge.created"
            | "workflow.created"
            | "experiment.created"
            | null
            | undefined) ?? undefined,
          undefined,
          max,
        );
        if (!cancelled) {
          setEvents(rows as WorkspaceEventRow[]);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void tickOnce();
    const id = setInterval(() => void tickOnce(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [max, tick, eventType]);

  if (loading && events.length === 0 && !error) {
    return <WorkbenchOperationState kind="loading" density="compact" skeletonRows={3} />;
  }
  if (error && events.length === 0) {
    return (
      <WorkbenchOperationState
        kind="error"
        density="compact"
        title="Could not load activity"
        detail={error}
        action={
          <WorkbenchRetryAction
            onClick={() => {
              setLoading(true);
              setTick((t) => t + 1);
            }}
          />
        }
      />
    );
  }
  if (events.length === 0) {
    return <WorkbenchOperationState kind="empty" density="compact" title={FEED_EMPTY_TEXT} />;
  }

  return (
    <ol className="space-y-1">
      {events.map((event) => {
        const visual = eventVisualFor(event.type);
        const Icon = visual.icon;
        return (
          <li
            key={event.id}
            className="flex items-start gap-3 rounded-control px-2 py-2 text-label transition-colors duration-(--motion-fast) ease-standard hover:bg-interactive/50"
          >
            <span
              aria-hidden="true"
              className={cn(
                "mt-1 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full",
                visual.dotClass,
              )}
            >
              <Icon className="h-2.5 w-2.5 text-background" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate font-medium text-foreground">
                  {visual.label}
                  <span className="ml-2 font-normal text-muted-foreground">· {event.actor}</span>
                </span>
                <span
                  className="shrink-0 tabular-nums text-micro text-muted-foreground"
                  title={formatTimestamp(event.created_at)}
                >
                  {formatRelative(event.created_at)}
                </span>
              </div>
              <span className="mt-1 flex flex-wrap gap-x-2 font-mono text-micro text-muted-foreground">
                {event.refs.map((ref) => {
                  const resolved = resolveEventRef(ref, event.type, knownRunIds, event.payload);
                  if (resolved.kind === "run") {
                    return (
                      <WorkbenchAction
                        kind="ghost"
                        size="content"
                        key={ref}
                        type="button"
                        className="truncate text-accent underline-offset-2 hover:underline"
                        onClick={() => onSelectRun(resolved.runId)}
                      >
                        {resolved.text}
                      </WorkbenchAction>
                    );
                  }
                  if (resolved.kind === "knowledge") {
                    return (
                      <WorkbenchAction
                        kind="ghost"
                        size="content"
                        key={ref}
                        type="button"
                        className="truncate text-accent underline-offset-2 hover:underline"
                        onClick={() => onOpenKnowledge(resolved.path)}
                      >
                        {resolved.text}
                      </WorkbenchAction>
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
