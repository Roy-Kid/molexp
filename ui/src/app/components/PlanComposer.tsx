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
const POLL_INTERVAL_MS = 1000;

interface PlanComposerProps {
  projectId: string;
  experimentId: string;
  /** Called once a plan completes — open session / refresh / switch tab. */
  onPlanComplete: (task: PlanTaskResponse) => void;
}

/**
 * PlanMode composer — UI counterpart to `molexp plan`.
 */
export function PlanComposer({ projectId, experimentId, onPlanComplete }: PlanComposerProps) {
  const [draft, setDraft] = useState("");
  const [task, setTask] = useState<PlanTaskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [execute, setExecute] = useState(false);
  const [computeTarget, setComputeTarget] = useState<string | null>(null);
  const [targetNames, setTargetNames] = useState<string[]>([]);

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
  const inFlight = isRunning || isWaitingApproval;

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
        setError(err instanceof Error ? err.message : "Poll failed.");
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
      setError(err instanceof Error ? err.message : "Failed to start.");
    }
  };

  const completed = task?.status === "completed";

  return (
    <div className="space-y-3">
      <Textarea
        placeholder="Describe the experiment…"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        disabled={inFlight}
        rows={3}
      />
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={execute}
            disabled={inFlight}
            onChange={(e) => setExecute(e.target.checked)}
            className="h-3.5 w-3.5 cursor-pointer rounded border border-border accent-primary"
          />
          Execute after approvals
        </label>
        {execute && targetNames.length > 0 && (
          <Select
            value={computeTarget ?? LOCAL_TARGET_VALUE}
            onValueChange={(v) => setComputeTarget(v === LOCAL_TARGET_VALUE ? null : v)}
            disabled={inFlight}
          >
            <SelectTrigger className="h-7 w-40 text-xs" aria-label="Target">
              <SelectValue placeholder="Target" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={LOCAL_TARGET_VALUE}>local</SelectItem>
              {targetNames.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {isWaitingApproval && taskId && <ApprovalsInbox taskId={taskId} />}
      {completed && <p className="text-xs text-success-foreground">Done — open Workflow tab.</p>}
      <div className="flex items-center justify-between gap-2">
        <span className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
          {task?.execute && <Badge variant="outline">execute</Badge>}
          {isWaitingApproval
            ? "Awaiting approval"
            : isRunning
              ? "Generating…"
              : task?.runId
                ? `Run ${task.runId}`
                : ""}
        </span>
        <Button size="sm" onClick={handleSubmit} disabled={!draft.trim() || inFlight}>
          <Sparkles className="h-3.5 w-3.5" />
          {isWaitingApproval ? "Waiting…" : isRunning ? "…" : "Plan"}
        </Button>
      </div>
    </div>
  );
}
