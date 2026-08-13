import { RefreshCw, Terminal } from "lucide-react";
import type { JSX } from "react";
import { useEffect, useRef } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  WorkbenchIconAction,
  WorkbenchOperationState,
  WorkbenchRetryAction,
} from "@/components/workbench";
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
        <span className="text-label text-muted-foreground">
          stdout / stderr · <span className="text-foreground">{attemptLabel}</span>
        </span>
        <div className="ml-auto flex items-center gap-1">
          {history.length > 1 && (
            <Select
              value={selectedExecutionId ?? "__latest__"}
              onValueChange={(value) => {
                onSelectExecution(value === "__latest__" ? null : value);
              }}
            >
              <SelectTrigger size="sm" className="max-w-36 text-micro" aria-label="Select attempt">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__latest__">Latest</SelectItem>
                {history.map((exec, index) => (
                  <SelectItem key={exec.executionId} value={exec.executionId}>
                    #{index + 1} · {exec.status}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <WorkbenchIconAction
            label="Refresh logs"
            kind="ghost"
            type="button"
            className="h-control-compact w-control-compact text-muted-foreground"
            onClick={onRefresh}
            aria-label="Refresh logs"
            title="Refresh logs"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "mol-motion-progress-spin")} />
          </WorkbenchIconAction>
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="min-h-0 flex-1 overflow-auto bg-muted/15 px-3 py-3 font-mono text-micro leading-relaxed"
      >
        {error ? (
          <WorkbenchOperationState
            kind="error"
            density="compact"
            title="Could not load logs"
            detail={error}
            action={<WorkbenchRetryAction onClick={onRefresh} />}
          />
        ) : loading && !logs ? (
          <WorkbenchOperationState kind="loading" density="compact" skeletonRows={4} />
        ) : logs ? (
          <div className="space-y-4">
            <LogBlock title="stdout" body={logs.stdout} tone="default" />
            <LogBlock title="stderr" body={logs.stderr} tone="error" />
          </div>
        ) : (
          <WorkbenchOperationState kind="empty" density="compact" title="No logs yet." />
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
    <div className="mb-1 text-micro font-medium text-muted-foreground">{title}</div>
    <pre
      className={cn(
        "whitespace-pre-wrap break-words",
        tone === "error" ? "text-status-failed-foreground" : "text-foreground",
        !body && "italic text-muted-foreground",
      )}
    >
      {body && body.length > 0 ? body : `No ${title} captured.`}
    </pre>
  </section>
);
