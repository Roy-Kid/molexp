/**
 * PlanCard — inline chat card for a structured ``PlanCreated`` event.
 *
 * Plan-mode in molexp emits ONE kind of plan: a runnable workflow.
 * Every numbered step in ``plan_markdown`` corresponds to one node in
 * ``workflow_preview.workflow_ir.task_configs``. The IR is the only
 * editable surface — the graph and Python preview are derived from it.
 * On approval the session flips out of plan mode and the agent proceeds
 * to bind / run the workflow. See
 * :data:`molexp.plugins.agent_pydanticai._pydantic_ai.system_prompt.PLAN_MODE_ADDENDUM`
 * for the agent-side contract.
 */

import { CheckCircle2, Code2, FilePen, Network, PlayCircle, Terminal, X } from "lucide-react";
import { type JSX, useCallback, useEffect, useMemo, useState } from "react";
import { agentApi } from "@/app/state/api";
import type { ApiSessionEvent } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { renderPythonFromIr } from "@/lib/workflow-python";
import { WorkflowGraph, type WorkflowIR } from "./WorkflowGraph";

// ─── Step parser ────────────────────────────────────────────────────────────

export interface PlanStep {
  index: number;
  toolName: string | null;
  args: string | null;
  rationale: string;
  raw: string;
}

const STEP_LINE_RE =
  /^\s*(?:[*-]|\d+\.)\s+(?:\*\*)?(?:Step\s+\d+:?\s*)?([a-zA-Z_][\w]*)?\s*(?:\(([^)]*)\))?(?:\*\*)?\s*(?:[—\-:]\s*(.+))?$/;

/** Parse the agent's ``plan_markdown`` into structured Step rows. */
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

// ─── Payload shapes ─────────────────────────────────────────────────────────

interface WorkflowPreviewPayload {
  mermaid?: string;
  python_script?: string;
  workflow_ir: WorkflowIR | Record<string, unknown>;
  intervention_points?: string[];
}

export interface PlanCreatedPayload {
  request_id: string;
  plan_markdown: string;
  workflow_preview: WorkflowPreviewPayload;
}

const isWorkflowIr = (value: unknown): value is WorkflowIR => {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return Array.isArray(v.task_configs) && Array.isArray(v.links);
};

// ─── Public component ───────────────────────────────────────────────────────

interface PlanCardProps {
  sessionId: string;
  event: ApiSessionEvent;
  /** Fired after the user approves or rejects so the parent can re-fetch. */
  onResolved?: () => void;
}

export const PlanCard = ({ sessionId, event, onResolved }: PlanCardProps): JSX.Element => {
  const payload = event.payload as Partial<PlanCreatedPayload> | undefined;

  // Defensive: a misbehaving agent (or a stale on-disk fixture) might
  // emit a PlanCreated event missing contract fields. Render a clear
  // error block instead of throwing — crashing the chat would make the
  // session unusable for the rest of the conversation.
  if (
    !payload ||
    typeof payload.request_id !== "string" ||
    typeof payload.plan_markdown !== "string" ||
    !payload.workflow_preview ||
    typeof payload.workflow_preview !== "object" ||
    !("workflow_ir" in payload.workflow_preview)
  ) {
    return <MalformedPlanEvent payload={payload} reason="missing required fields" />;
  }

  return (
    <WorkflowPlanCard
      sessionId={sessionId}
      requestId={payload.request_id}
      planMarkdown={payload.plan_markdown}
      workflowPreview={payload.workflow_preview}
      onResolved={onResolved}
    />
  );
};

// ─── Workflow card ──────────────────────────────────────────────────────────

interface WorkflowPlanCardProps {
  sessionId: string;
  requestId: string;
  planMarkdown: string;
  workflowPreview: WorkflowPreviewPayload;
  onResolved?: () => void;
}

type ViewMode = "graph" | "ir" | "python";

const WorkflowPlanCard = ({
  sessionId,
  requestId,
  planMarkdown,
  workflowPreview,
  onResolved,
}: WorkflowPlanCardProps): JSX.Element => {
  const ir = workflowPreview.workflow_ir;
  const initialIr = useMemo(
    () => (ir && typeof ir === "object" ? JSON.stringify(ir, null, 2) : ""),
    [ir],
  );
  const initialPython = useMemo(
    () => workflowPreview.python_script ?? "",
    [workflowPreview.python_script],
  );

  const decision = usePlanDecision({
    sessionId,
    requestId,
    planMarkdown,
    initialIr,
    onResolved,
  });

  const [viewMode, setViewMode] = useState<ViewMode>("graph");

  const steps = useMemo(() => parsePlan(decision.planText), [decision.planText]);

  // Always derive the graph from the live IR text so edits propagate to
  // both the graph and the Python preview without an extra "apply" step.
  // Falls back to the original ``ir`` if the textarea content is not
  // valid JSON yet (mid-edit).
  const graphIr = useMemo<WorkflowIR | null>(() => {
    const fromText = safeParse(decision.irText);
    if (isWorkflowIr(fromText)) return fromText;
    return isWorkflowIr(ir) ? ir : null;
  }, [decision.irText, ir]);

  const interventionPoints = workflowPreview.intervention_points ?? [];

  // Python is auto-derived from the IR — edits to the IR re-render it
  // immediately. Server-provided ``python_script`` is used as a fallback
  // only when the IR fails to parse (so the user still sees something).
  const pythonText = useMemo(() => {
    if (!graphIr) return initialPython;
    try {
      return renderPythonFromIr(graphIr);
    } catch {
      return initialPython;
    }
  }, [graphIr, initialPython]);

  return (
    <div className="rounded-xl border border-violet-300 bg-card p-3 shadow-sm dark:border-violet-700">
      <div className="mb-3 flex items-center gap-2">
        <Badge className="border-transparent bg-violet-600 text-white hover:bg-violet-600 dark:bg-violet-500 dark:hover:bg-violet-500">
          <PlayCircle className="mr-1 h-3 w-3" />
          Workflow plan
        </Badge>
      </div>

      <StepPlanSection
        accentClass="bg-violet-600 text-white"
        steps={steps}
        planText={decision.planText}
      />

      <section className="mb-3 rounded-lg border border-border/60 bg-background p-3 shadow-sm">
        <header className="mb-2 flex items-center gap-2">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Workflow
          </h4>
          {graphIr ? (
            <Badge variant="outline" className="text-[10px]">
              {graphIr.task_configs.length} task
              {graphIr.task_configs.length === 1 ? "" : "s"} · {graphIr.links.length} link
              {graphIr.links.length === 1 ? "" : "s"}
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[10px] text-amber-600">
              malformed IR
            </Badge>
          )}
          <div className="ml-auto inline-flex rounded-md border border-border/60 bg-background p-0.5 text-[11px]">
            <ViewToggle
              icon={Network}
              label="Graph"
              active={viewMode === "graph"}
              onClick={() => setViewMode("graph")}
            />
            <ViewToggle
              icon={FilePen}
              label="IR"
              active={viewMode === "ir"}
              onClick={() => setViewMode("ir")}
            />
            <ViewToggle
              icon={Code2}
              label="Python"
              active={viewMode === "python"}
              onClick={() => setViewMode("python")}
            />
          </div>
        </header>

        {viewMode === "graph" &&
          (graphIr ? (
            <WorkflowGraph ir={graphIr} className="mb-2" />
          ) : (
            <p className="mb-2 rounded-md border border-dashed border-amber-300 bg-amber-50/40 px-3 py-2 text-[11px] text-amber-700 dark:border-amber-700 dark:bg-amber-950/30 dark:text-amber-300">
              The agent's IR did not parse as a valid workflow (need ``task_configs`` and
              ``links``). Switch to the IR view to repair the JSON before approving.
            </p>
          ))}

        {viewMode === "ir" && (
          <div>
            <div className="mb-2 flex items-center justify-between gap-2">
              <p className="text-[11px] text-muted-foreground">
                Edit the IR — graph and Python regenerate automatically.
              </p>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-[11px]"
                onClick={() => decision.setEditingIr((v) => !v)}
                disabled={decision.submitting || decision.resolved}
              >
                <FilePen className="mr-1 h-3 w-3" />
                {decision.editingIr ? "Done" : "Edit"}
              </Button>
            </div>
            {decision.editingIr ? (
              <>
                <Textarea
                  value={decision.irText}
                  onChange={(e) => {
                    decision.setIrText(e.target.value);
                    if (decision.irError) decision.setIrError(null);
                  }}
                  rows={Math.min(20, Math.max(6, decision.irText.split("\n").length))}
                  className={
                    "font-mono text-[11px] " +
                    (decision.irError ? "border-destructive focus-visible:ring-destructive" : "")
                  }
                />
                {decision.irError && (
                  <p className="mt-1 text-[11px] text-destructive">{decision.irError}</p>
                )}
              </>
            ) : (
              <pre className="max-h-72 overflow-auto rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-[11px]">
                {decision.irText}
              </pre>
            )}
          </div>
        )}

        {viewMode === "python" &&
          (pythonText ? (
            <pre className="max-h-72 overflow-auto rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-[11px]">
              {pythonText}
            </pre>
          ) : (
            <p className="mb-2 rounded-md border border-dashed border-border bg-muted/20 px-3 py-2 text-[11px] text-muted-foreground">
              The IR is not valid JSON — fix it in the IR tab and the Python preview will regenerate
              automatically.
            </p>
          ))}

        {interventionPoints.length > 0 && (
          <div className="mt-2 rounded-md bg-muted/20 px-3 py-2">
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Intervene here
            </p>
            <ul className="ml-4 list-disc space-y-0.5 text-[11px] text-muted-foreground">
              {interventionPoints.map((point) => (
                <li key={`intv-${point.slice(0, 60)}`}>{point}</li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <DecisionPanel
        decision={decision}
        edited={decision.irEdited}
        approveLabel={decision.irEdited ? "Approve with edits" : "Approve & execute"}
        approveHint="Approve to let the agent bind + run this workflow."
      />
    </div>
  );
};

const ViewToggle = ({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: typeof Network;
  label: string;
  active: boolean;
  onClick: () => void;
}): JSX.Element => (
  <button
    type="button"
    onClick={onClick}
    className={
      "inline-flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-medium transition-colors " +
      (active
        ? "bg-violet-500/15 text-violet-700 dark:text-violet-300"
        : "text-muted-foreground hover:text-foreground")
    }
    aria-pressed={active}
  >
    <Icon className="h-3 w-3" />
    {label}
  </button>
);

// ─── Shared decision state hook ─────────────────────────────────────────────

interface UsePlanDecisionArgs {
  sessionId: string;
  requestId: string;
  planMarkdown: string;
  initialIr: string | null;
  onResolved?: () => void;
}

interface PlanDecisionState {
  planText: string;
  irText: string;
  setIrText: (s: string) => void;
  editingIr: boolean;
  setEditingIr: (fn: (v: boolean) => boolean) => void;
  irError: string | null;
  setIrError: (s: string | null) => void;
  rejecting: boolean;
  setRejecting: (v: boolean) => void;
  feedback: string;
  setFeedback: (s: string) => void;
  submitting: boolean;
  error: string | null;
  resolved: boolean;
  irEdited: boolean;
  approve: () => Promise<void>;
  reject: () => Promise<void>;
}

const usePlanDecision = (args: UsePlanDecisionArgs): PlanDecisionState => {
  const { sessionId, requestId, planMarkdown, initialIr, onResolved } = args;

  const [irText, setIrText] = useState(initialIr ?? "");
  const [editingIr, setEditingIr] = useState(false);
  const [irError, setIrError] = useState<string | null>(null);

  const [rejecting, setRejecting] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resolved, setResolved] = useState(false);

  // Re-seed editable state when the agent emits a new plan in the same session
  // (e.g. after the user rejected the first one).
  useEffect(() => {
    setResolved(false);
    setError(null);
    setRejecting(false);
    setFeedback("");
  }, [planMarkdown, requestId]);
  useEffect(() => {
    setIrText(initialIr ?? "");
    setEditingIr(false);
    setIrError(null);
  }, [initialIr]);

  const irEdited = initialIr !== null && irText.trim() !== (initialIr ?? "").trim();

  const approve = useCallback(async (): Promise<void> => {
    setError(null);
    setIrError(null);
    let editedIrParsed: Record<string, unknown> | null = null;
    if (irEdited) {
      try {
        editedIrParsed = JSON.parse(irText) as Record<string, unknown>;
      } catch (err) {
        setIrError(`Workflow IR is not valid JSON: ${(err as Error).message}`);
        return;
      }
    }
    setSubmitting(true);
    try {
      await agentApi.respondPlan(sessionId, {
        requestId,
        approved: true,
        editedPlan: null,
        editedWorkflowIr: editedIrParsed,
      });
      setResolved(true);
      onResolved?.();
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }, [irEdited, irText, onResolved, requestId, sessionId]);

  const reject = useCallback(async (): Promise<void> => {
    setError(null);
    setSubmitting(true);
    try {
      await agentApi.respondPlan(sessionId, {
        requestId,
        approved: false,
        feedback: feedback.trim(),
      });
      setResolved(true);
      onResolved?.();
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }, [feedback, onResolved, requestId, sessionId]);

  return {
    planText: planMarkdown,
    irText,
    setIrText,
    editingIr,
    setEditingIr,
    irError,
    setIrError,
    rejecting,
    setRejecting,
    feedback,
    setFeedback,
    submitting,
    error,
    resolved,
    irEdited,
    approve,
    reject,
  };
};

// ─── Shared sub-components ──────────────────────────────────────────────────

interface StepPlanSectionProps {
  accentClass: string;
  steps: PlanStep[];
  planText: string;
}

const StepPlanSection = ({ accentClass, steps, planText }: StepPlanSectionProps): JSX.Element => (
  <section className="mb-3 rounded-lg border border-border/60 bg-background p-3 shadow-sm">
    <header className="mb-2 flex items-center gap-2">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Step plan
      </h4>
      <Badge variant="outline" className="text-[10px]">
        {steps.length} step{steps.length === 1 ? "" : "s"}
      </Badge>
    </header>
    {steps.length > 0 ? (
      <ol className="flex flex-col gap-1.5">
        {steps.map((step) => (
          <li
            key={step.index}
            className="flex items-start gap-2 rounded-md bg-muted/30 px-2 py-1.5"
          >
            <span
              className={
                "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold " +
                accentClass
              }
            >
              {step.index}
            </span>
            <div className="min-w-0 flex-1">
              {step.toolName && (
                <div className="flex items-center gap-1.5 font-mono text-[11px]">
                  <Terminal className="h-3 w-3 text-blue-500" />
                  <span>{step.toolName}</span>
                  {step.args && <span className="text-muted-foreground">({step.args})</span>}
                </div>
              )}
              {step.rationale && (
                <p className="mt-0.5 text-[11px] text-muted-foreground">{step.rationale}</p>
              )}
              {!step.toolName && !step.rationale && (
                <p className="font-mono text-[11px]">{step.raw}</p>
              )}
            </div>
          </li>
        ))}
      </ol>
    ) : (
      <pre className="whitespace-pre-wrap rounded-md bg-muted/30 p-2 font-mono text-[11px]">
        {planText}
      </pre>
    )}
  </section>
);

interface DecisionPanelProps {
  decision: PlanDecisionState;
  edited: boolean;
  approveLabel: string;
  approveHint: string;
}

const DecisionPanel = ({
  decision,
  edited,
  approveLabel,
  approveHint,
}: DecisionPanelProps): JSX.Element => {
  if (decision.resolved) {
    return (
      <p className="rounded-md bg-muted/30 px-3 py-2 text-[11px] italic text-muted-foreground">
        Decision recorded — the agent has resumed.
      </p>
    );
  }
  if (decision.rejecting) {
    return (
      <section className="rounded-lg border border-amber-200 bg-amber-50/60 p-3 dark:border-amber-800 dark:bg-amber-950/20">
        <p className="mb-2 text-[11px] font-medium text-amber-800 dark:text-amber-300">
          Tell the agent what to revise:
        </p>
        <Textarea
          value={decision.feedback}
          onChange={(e) => decision.setFeedback(e.target.value)}
          placeholder="e.g. scope this to project foo only; replace step 3 with list_runs"
          rows={3}
          className="text-[11px]"
          disabled={decision.submitting}
        />
        {decision.error && <p className="mt-2 text-[11px] text-destructive">{decision.error}</p>}
        <div className="mt-2 flex justify-end gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => decision.setRejecting(false)}
            disabled={decision.submitting}
          >
            Back
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={() => void decision.reject()}
            disabled={decision.submitting}
          >
            <X className="mr-1 h-3.5 w-3.5" />
            Reject plan
          </Button>
        </div>
      </section>
    );
  }
  return (
    <section className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/60 bg-card px-3 py-2 shadow-sm">
      <div className="text-[11px] text-muted-foreground">
        {edited ? "Edits will be sent with the approval." : approveHint}
      </div>
      <div className="flex gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => decision.setRejecting(true)}
          disabled={decision.submitting}
        >
          <X className="mr-1 h-3.5 w-3.5" />
          Reject
        </Button>
        <Button size="sm" onClick={() => void decision.approve()} disabled={decision.submitting}>
          {decision.submitting ? (
            <CheckCircle2 className="mr-1 h-3.5 w-3.5 animate-pulse" />
          ) : (
            <PlayCircle className="mr-1 h-3.5 w-3.5" />
          )}
          {approveLabel}
        </Button>
      </div>
      {decision.error && (
        <p className="basis-full text-[11px] text-destructive">{decision.error}</p>
      )}
    </section>
  );
};

const safeParse = (text: string): unknown => {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
};

const MalformedPlanEvent = ({
  payload,
  reason,
}: {
  payload: unknown;
  reason: string;
}): JSX.Element => (
  <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
    <p className="font-semibold">Malformed plan event</p>
    <p className="mt-1 text-muted-foreground">{reason}. Raw payload:</p>
    <pre className="mt-2 max-h-40 overflow-auto rounded bg-muted/60 p-2 font-mono text-[10px] text-foreground">
      {JSON.stringify(payload, null, 2)}
    </pre>
  </div>
);
