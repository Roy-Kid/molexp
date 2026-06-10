import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import type { JSX } from "react";

import { type ToolCallState, toolCallDurationSeconds } from "@/app/renderers/agentEvents";
import { formatDurationCompact } from "@/lib/format-time";

/**
 * One tool-call row that transitions started → completed in place.
 *
 * Mirrors the CLI renderer's vocabulary: a spinning info-tinted row while
 * running, then `✓ name · summary · 0.8s` (success) or a destructive `✗`
 * row on failure. State is carried by icon + color + text, never color
 * alone. The same row updates in place (the fold keys started/completed
 * to one `ToolCallState`).
 */
export const ToolCallRow = ({ call }: { call: ToolCallState }): JSX.Element => {
  if (call.status === "started") {
    return (
      <div className="flex min-w-0 items-center gap-2 rounded-sm px-1.5 py-1 text-xs">
        <Loader2 className="h-3.5 w-3.5 flex-none animate-spin text-info" />
        <span className="truncate font-mono text-foreground">
          {call.toolName}
          {call.argsSummary && <span className="text-muted-foreground">({call.argsSummary})</span>}
        </span>
        <span className="flex-none text-muted-foreground">running…</span>
      </div>
    );
  }

  const failed = call.ok === false;
  const Icon = failed ? XCircle : CheckCircle2;
  const duration = formatDurationCompact(toolCallDurationSeconds(call));

  return (
    <div className="group/tool flex min-w-0 items-center gap-2 rounded-sm px-1.5 py-1 text-xs transition-colors hover:bg-muted/40">
      <Icon
        className={`h-3.5 w-3.5 flex-none ${failed ? "text-destructive" : "text-success"}`}
        aria-label={failed ? "Tool call failed" : "Tool call succeeded"}
      />
      <span className={`truncate font-mono ${failed ? "text-destructive" : "text-foreground"}`}>
        {call.toolName}
      </span>
      {call.resultSummary && (
        <span
          className={`min-w-0 truncate ${failed ? "text-destructive/90" : "text-muted-foreground"}`}
          title={call.resultSummary}
        >
          {call.resultSummary}
        </span>
      )}
      {duration && (
        <span className="ml-auto flex-none font-mono text-[10px] tabular-nums text-muted-foreground/70">
          {duration}
        </span>
      )}
    </div>
  );
};
