/**
 * PlanView — center-panel renderer for plan-mode sessions.
 *
 * A plan-mode session ends with a structured execution plan in its
 * final assistant message. This component parses that message and
 * presents the steps with explicit Execute / Edit affordances so the
 * user can promote the plan into a regular session.
 */

import { CheckCircle2, FilePen, PlayCircle, Sparkles, Terminal } from "lucide-react";
import { type JSX, useCallback, useMemo, useState } from "react";
import { agentApi, planApi } from "@/app/state/api";
import type { ApiAgentSession, ApiSessionEvent, WorkspaceSnapshot } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";

export interface PlanStep {
  index: number;
  toolName: string | null;
  args: string | null;
  rationale: string;
  raw: string;
}

interface PlanViewProps {
  session: ApiAgentSession;
  events: ApiSessionEvent[];
  snapshot: WorkspaceSnapshot;
  onPromoted: (newSessionId: string) => void;
}

// Match `1. tool_name(args) — rationale` (or `--`, or no rationale at all).
// Group 1: index, group 2: tool name, group 3: args, group 4: rationale
const STEP_LINE_RE =
  /^\s*(?:[*-]|\d+\.)\s+(?:\*\*)?(?:Step\s+\d+:?\s*)?([a-zA-Z_][\w]*)?\s*(?:\(([^)]*)\))?(?:\*\*)?\s*(?:[—\-:]\s*(.+))?$/;

export const parsePlan = (markdown: string): PlanStep[] => {
  const out: PlanStep[] = [];
  let counter = 0;
  for (const line of markdown.split("\n")) {
    const match = STEP_LINE_RE.exec(line);
    if (!match) continue;
    const [_, tool, args, rationale] = match;
    if (!tool && !args && !rationale) continue;
    counter += 1;
    out.push({
      index: counter,
      toolName: tool ? tool.trim() : null,
      args: args ? args.trim() : null,
      rationale: rationale ? rationale.trim() : "",
      raw: line.trim(),
    });
  }
  return out;
};

const extractFinalSummary = (session: ApiAgentSession, events: ApiSessionEvent[]): string => {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event.type !== "SessionCompletedEvent") continue;
    const summary = event.payload?.summary;
    if (typeof summary === "string") return summary;
  }
  // Fallback: scan the original events buffer attached to the session.
  for (let i = (session.events ?? []).length - 1; i >= 0; i -= 1) {
    const event = (session.events ?? [])[i];
    if (event.type !== "SessionCompletedEvent") continue;
    const summary = event.payload?.summary;
    if (typeof summary === "string") return summary;
  }
  return "";
};

export const PlanView = ({ session, events, onPromoted }: PlanViewProps): JSX.Element => {
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const summary = useMemo(() => extractFinalSummary(session, events), [session, events]);
  const steps = useMemo(() => parsePlan(summary), [summary]);
  const isTerminal = session.status === "completed" || session.status === "failed";

  const handleEdit = useCallback(() => {
    setEditText(summary);
    setEditing(true);
  }, [summary]);

  const handleExecute = useCallback(async () => {
    setSubmitting(true);
    setError(null);
    try {
      let nextSessionId: string;
      if (editing && editText.trim() !== summary.trim()) {
        // Edited plan: start a brand-new session that uses the edited
        // plan as the system prompt override. Reuse the goal description.
        const created = await agentApi.createSession(session.goalDescription, [], {
          instructionsOverride:
            "An approved execution plan follows. Execute it step-by-step using " +
            "the available tools, then report the outcome.\n\n" +
            editText.trim(),
          skillId: session.skillId ?? undefined,
        });
        nextSessionId = created.sessionId;
      } else {
        const created = await planApi.execute(session.sessionId);
        nextSessionId = created.sessionId;
      }
      onPromoted(nextSessionId);
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }, [
    editing,
    editText,
    onPromoted,
    session.goalDescription,
    session.skillId,
    session.sessionId,
    summary,
  ]);

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2 border-b border-border/70 bg-violet-500/5 px-4 py-2 text-sm md:px-8">
        <Sparkles className="h-4 w-4 text-violet-500" />
        <span className="font-medium">Plan mode session</span>
        <Badge variant="outline" className="text-[10px]">
          read-only
        </Badge>
        <span className="ml-auto text-[11px] text-muted-foreground">
          {steps.length > 0
            ? `${steps.length} step${steps.length === 1 ? "" : "s"} parsed`
            : isTerminal
              ? "no plan parsed — review raw output below"
              : "agent is composing the plan…"}
        </span>
      </div>

      <ScrollArea className="flex-1">
        <div className="mx-auto flex max-w-3xl flex-col gap-3 px-4 py-6 md:px-8">
          {!isTerminal && (
            <p className="rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
              The agent is exploring the workspace and will emit a structured plan shortly. The
              Execute button activates once the plan is ready.
            </p>
          )}

          {steps.length > 0 && !editing && (
            <ol className="flex flex-col gap-2">
              {steps.map((step) => (
                <li
                  key={step.index}
                  className="rounded-lg border border-border/70 bg-card px-3 py-2 shadow-sm"
                >
                  <div className="flex items-start gap-2">
                    <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-violet-500/10 text-[10px] font-semibold text-violet-700">
                      {step.index}
                    </span>
                    <div className="flex-1 min-w-0">
                      {step.toolName && (
                        <div className="flex items-center gap-1.5 font-mono text-xs">
                          <Terminal className="h-3 w-3 text-blue-500" />
                          <span>{step.toolName}</span>
                          {step.args && (
                            <span className="text-muted-foreground">({step.args})</span>
                          )}
                        </div>
                      )}
                      {step.rationale && (
                        <p className="mt-1 text-xs text-muted-foreground">{step.rationale}</p>
                      )}
                      {!step.toolName && !step.rationale && (
                        <p className="font-mono text-xs">{step.raw}</p>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          )}

          {steps.length === 0 && summary && !editing && (
            <pre className="whitespace-pre-wrap rounded-md border border-border/70 bg-card p-3 font-mono text-xs">
              {summary}
            </pre>
          )}

          {editing && (
            <Textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              rows={Math.max(8, editText.split("\n").length + 1)}
              className="font-mono text-xs"
            />
          )}

          {error && <p className="text-xs text-destructive">{error}</p>}
        </div>
      </ScrollArea>

      <div className="sticky bottom-0 border-t border-border/70 bg-background/95 px-4 py-2 backdrop-blur md:px-8">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-2">
          <p className="text-[11px] text-muted-foreground">
            {isTerminal
              ? "Review the plan, then execute or edit before launching."
              : "Plan will appear when the agent completes."}
          </p>
          <div className="flex gap-2">
            {editing ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setEditing(false)}
                disabled={submitting}
              >
                Cancel edit
              </Button>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={handleEdit}
                disabled={!isTerminal || submitting || !summary}
                title="Edit the plan before executing"
              >
                <FilePen className="mr-1 h-3.5 w-3.5" /> Edit plan
              </Button>
            )}
            <Button
              size="sm"
              onClick={() => void handleExecute()}
              disabled={!isTerminal || submitting || !summary}
              title="Start a follow-up session that executes this plan"
            >
              {submitting ? (
                <CheckCircle2 className="mr-1 h-3.5 w-3.5 animate-pulse" />
              ) : (
                <PlayCircle className="mr-1 h-3.5 w-3.5" />
              )}
              Execute plan
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
