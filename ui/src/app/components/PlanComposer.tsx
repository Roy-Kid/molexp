import { Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import type { PlanTaskResponse } from "@/api/generated/models/PlanTaskResponse";
import { TargetsService } from "@/api/generated/services/TargetsService";
import { ApprovalsInbox } from "@/app/renderers/agent/ApprovalsInbox";
import { workspaceApi } from "@/app/state/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

const LOCAL_TARGET_VALUE = "__local__";
const EXECUTE_HELP_COPY =
  "Runs the generated workflow on the server host after all three approval gates (experiment spec, plan review, execution review) are granted. Off: the plan ends at the descriptive execution report.";
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
  const [execute, setExecute] = useState(false);
  const [computeTarget, setComputeTarget] = useState<string | null>(null);
  const [targetNames, setTargetNames] = useState<string[]>([]);

  // Known compute targets feed the (optional) descriptive-target select; an
  // empty registry simply means the local default — the select stays hidden.
  useEffect(() => {
    let cancelled = false;
    TargetsService.listTargetsEndpointApiTargetsGet()
      .then((res) => {
        if (!cancelled) setTargetNames(res.targets.map((t) => t.name));
      })
      .catch(() => {
        if (!cancelled) setTargetNames([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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
      const started = await workspaceApi.createPlanTask(projectId, experimentId, {
        draft: text,
        execute,
        compute_target: computeTarget,
      });
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
      <div className="space-y-1.5 rounded-md border border-border/60 px-3 py-2">
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={execute}
            disabled={inFlight}
            onChange={(e) => setExecute(e.target.checked)}
            className="h-3.5 w-3.5 cursor-pointer rounded border border-border accent-primary"
          />
          <span className="font-medium">Execute after approvals</span>
        </label>
        <p className="text-xs text-muted-foreground">{EXECUTE_HELP_COPY}</p>
        {execute && targetNames.length > 0 && (
          <Select
            value={computeTarget ?? LOCAL_TARGET_VALUE}
            onValueChange={(v) => setComputeTarget(v === LOCAL_TARGET_VALUE ? null : v)}
            disabled={inFlight}
          >
            <SelectTrigger className="h-8 w-full text-xs" aria-label="Compute target">
              <SelectValue placeholder="Compute target (report only)" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={LOCAL_TARGET_VALUE}>local default</SelectItem>
              {targetNames.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {execute && targetNames.length === 0 && (
          <p className="text-xs text-muted-foreground">
            No compute targets registered — the execution report describes the local default.
          </p>
        )}
      </div>
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
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          {task?.execute && <Badge variant="outline">execute</Badge>}
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
          {isWaitingApproval
            ? "Waiting for approval…"
            : isRunning
              ? "Generating…"
              : "Generate plan"}
        </Button>
      </div>
    </div>
  );
}
