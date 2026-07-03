import { Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import type { PlanTaskResponse } from "@/api/generated/models/PlanTaskResponse";
import { ApprovalsInbox } from "@/app/renderers/agent/ApprovalsInbox";
import { workspaceApi } from "@/app/state/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

const POLL_INTERVAL_MS = 1000;

interface PlanComposerProps {
  projectId: string;
  experimentId: string;
  /** Called once a plan completes, with the finished task — so the caller can
   * open the plan's session (its `taskId`) and/or refresh the snapshot. */
  onPlanComplete: (task: PlanTaskResponse) => void;
}

/**
 * "Generate plan with AI" composer: a draft textarea + button that starts a
 * PlanMode background task (POST), polls its status until terminal, and on
 * completion refreshes the workspace so the generated workflow renders. The UI
 * counterpart to the `molexp plan` CLI.
 */
export function PlanComposer({ projectId, experimentId, onPlanComplete }: PlanComposerProps) {
  const [draft, setDraft] = useState("");
  const [task, setTask] = useState<PlanTaskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const taskId = task?.taskId ?? null;
  const isRunning = task?.status === "running";
  const isWaitingApproval = task?.status === "waiting_approval";
  // Waiting for an approval decision is still "in flight": keep polling so the
  // resumed pipeline's progress lands here without a manual refresh.
  const inFlight = isRunning || isWaitingApproval;

  // Poll the in-flight task until it reaches a terminal status.
  useEffect(() => {
    if (!taskId || !inFlight) return;
    let cancelled = false;
    const handle = window.setInterval(async () => {
      try {
        const next = await workspaceApi.getPlanTask(projectId, experimentId, taskId);
        if (cancelled) return;
        setTask(next);
        if (next.status === "completed") {
          onPlanComplete(next);
        } else if (next.status === "failed" || next.status === "cancelled") {
          setError(next.error ?? `Plan ${next.status}.`);
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to poll plan status.");
        setTask((prev) => (prev ? { ...prev, status: "failed" } : prev));
      }
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [taskId, inFlight, projectId, experimentId, onPlanComplete]);

  const handleSubmit = async () => {
    const text = draft.trim();
    if (!text || inFlight) return;
    setError(null);
    try {
      const started = await workspaceApi.createPlanTask(projectId, experimentId, { draft: text });
      setTask(started);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start plan.");
    }
  };

  const completed = task?.status === "completed";

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Describe the experiment in plain language. PlanMode drafts a report, compiles a workflow,
        and grounds each step against the molcrafts toolchain.
      </p>
      <Textarea
        placeholder="e.g. Build a coarse-grained zwitterionic polymer melt and measure its conductivity…"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        disabled={inFlight}
        rows={4}
      />
      {error && <p className="text-sm text-red-500">{error}</p>}
      {isWaitingApproval && taskId && (
        // The pipeline suspended at an approval gate: decide inline (same
        // cards + decision path as the hub-level approvals inbox).
        <ApprovalsInbox taskId={taskId} />
      )}
      {completed && (
        <p className="text-sm text-emerald-600">
          Plan complete — open the Workflow tab to view the generated graph.
        </p>
      )}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {isWaitingApproval
            ? "Waiting for your approval…"
            : isRunning
              ? `Generating… (${task?.status})`
              : task?.runId
                ? `Run ${task.runId}`
                : ""}
        </span>
        <Button onClick={handleSubmit} disabled={!draft.trim() || inFlight}>
          <Sparkles className="h-4 w-4" />
          {isWaitingApproval ? "Waiting for approval…" : isRunning ? "Generating…" : "Generate plan"}
        </Button>
      </div>
    </div>
  );
}
