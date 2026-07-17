import { RefreshCw, Terminal } from "lucide-react";
import type { JSX } from "react";
import { useEffect, useRef } from "react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { WorkspaceRunRow } from "../types";
import type { RunLogsPayload } from "../useRunInspectorLogs";

interface RunInspectorLogsProps {
  run: WorkspaceRunRow;
  selectedExecutionId: string | null;
  onSelectExecution: (id: string | null) => void;
  logs: RunLogsPayload | null;
  error: string | null;
  loading: boolean;
  onRefresh: () => void;
}

export const RunInspectorLogs = ({
  run,
  selectedExecutionId,
  onSelectExecution,
  logs,
  error,
  loading,
  onRefresh,
}: RunInspectorLogsProps): JSX.Element => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const followRef = useRef(true);

  const history = run.executions;
  const effectiveId =
    selectedExecutionId ?? logs?.executionId ?? history[history.length - 1]?.executionId ?? null;
  const effectiveIndex = effectiveId ? history.findIndex((e) => e.executionId === effectiveId) : -1;
  const attemptLabel =
    effectiveIndex >= 0
      ? `#${effectiveIndex + 1}`
      : effectiveId
        ? effectiveId.slice(0, 12)
        : "latest";

  // Keep the viewport pinned to the bottom while the operator is "following"
  // the live tail (scroll near bottom). Manual scroll-up freezes follow.
  // Re-run when log text changes — deps are triggers, not read in the body.
  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll on log updates
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !followRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [logs?.stdout, logs?.stderr]);

  const handleScroll = (): void => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    followRef.current = distanceFromBottom < 48;
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-border/60 px-3 py-2">
        <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">
          stdout / stderr · <span className="text-foreground">{attemptLabel}</span>
        </span>
        <div className="ml-auto flex items-center gap-1">
          {history.length > 1 && (
            <select
              className="h-7 max-w-[140px] rounded-md border border-border bg-background px-1.5 text-[11px] text-foreground"
              value={selectedExecutionId ?? ""}
              onChange={(event) => {
                const value = event.target.value;
                onSelectExecution(value === "" ? null : value);
              }}
              aria-label="Select attempt"
            >
              <option value="">Latest</option>
              {history.map((exec, index) => (
                <option key={exec.executionId} value={exec.executionId}>
                  #{index + 1} · {exec.status}
                </option>
              ))}
            </select>
          )}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground"
            onClick={onRefresh}
            aria-label="Refresh logs"
            title="Refresh logs"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          </Button>
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="min-h-0 flex-1 overflow-auto bg-muted/15 px-3 py-3 font-mono text-[11px] leading-relaxed"
      >
        {error ? (
          <p className="text-destructive">{error}</p>
        ) : loading && !logs ? (
          <div className="space-y-2">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-[90%]" />
            <Skeleton className="h-3 w-[70%]" />
          </div>
        ) : logs ? (
          <div className="space-y-4">
            <LogBlock title="stdout" body={logs.stdout} tone="default" />
            <LogBlock title="stderr" body={logs.stderr} tone="error" />
          </div>
        ) : (
          <p className="text-muted-foreground">No logs yet.</p>
        )}
      </div>
    </div>
  );
};

const LogBlock = ({
  title,
  body,
  tone,
}: {
  title: string;
  body: string | null;
  tone: "default" | "error";
}): JSX.Element => (
  <section>
    <div className="mb-1 text-[11px] font-medium text-muted-foreground">{title}</div>
    <pre
      className={cn(
        "whitespace-pre-wrap break-words",
        tone === "error" ? "text-destructive/90" : "text-foreground",
        !body && "italic text-muted-foreground",
      )}
    >
      {body && body.length > 0 ? body : `No ${title} captured.`}
    </pre>
  </section>
);
