import { CheckCircle2, Loader2, XCircle } from "lucide-react";

import type { ToolCallState } from "@/app/renderers/agentEvents";

/**
 * One tool-call row that transitions started → completed in place (spec 03).
 *
 * Started: a spinning `⚙ name(args)`. Completed: `✓`/`✗ name — summary`,
 * mirroring the CLI `_render_tool_call_*` intent. The same row updates in
 * place (the fold keys started/completed to one `ToolCallState`).
 */
export const ToolCallRow = ({ call }: { call: ToolCallState }): JSX.Element => {
  if (call.status === "started") {
    return (
      <div className="flex items-center gap-2 text-xs text-blue-500">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span className="font-mono">
          ⚙ {call.toolName}
          {call.argsSummary ? `(${call.argsSummary})` : "()"}
        </span>
      </div>
    );
  }
  const Icon = call.ok ? CheckCircle2 : XCircle;
  return (
    <div
      className={`flex items-center gap-2 text-xs ${call.ok ? "text-green-600" : "text-red-500"}`}
    >
      <Icon className="h-3 w-3" />
      <span className="font-mono">
        {call.toolName}
        {call.resultSummary ? ` — ${call.resultSummary}` : ""}
      </span>
    </div>
  );
};
