/**
 * NL workspace curation — POST curate-tasks → poll → ApprovalsInbox.
 */

import { useEffect, useState } from "react";
import type { CurateTaskResponse } from "@/api/generated/models/CurateTaskResponse";
import { CurateTasksService } from "@/api/generated/services/CurateTasksService";
import { ApprovalsInbox } from "@/app/renderers/agent/ApprovalsInbox";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/components/ui/toast";
import { WorkbenchAction, WorkbenchTag } from "@/components/workbench";

const POLL_INTERVAL_MS = 1000;

interface CurateComposerProps {
  projectId: string;
  experimentId: string;
  onComplete?: (task: CurateTaskResponse) => void;
}

export function CurateComposer({
  projectId,
  experimentId,
  onComplete,
}: CurateComposerProps): JSX.Element {
  const [request, setRequest] = useState("");
  const [task, setTask] = useState<CurateTaskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const taskId = task?.taskId ?? null;
  const isRunning = task?.status === "running";
  const isWaitingApproval = task?.status === "waiting_approval";
  const inFlight = isRunning || isWaitingApproval;

  useEffect(() => {
    if (!taskId || !inFlight) return;
    let cancelled = false;
    const handle = window.setInterval(async () => {
      try {
        const next =
          await CurateTasksService.getCurateTaskApiProjectsProjectIdExperimentsExperimentIdCurateTasksTaskIdGet(
            projectId,
            experimentId,
            taskId,
          );
        if (cancelled) return;
        setTask(next);
        if (next.status === "completed") {
          toast.success("Curated");
          onComplete?.(next);
        } else if (next.status === "failed" || next.status === "cancelled") {
          setError(next.error ?? `Curate ${next.status}.`);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [taskId, inFlight, projectId, experimentId, onComplete]);

  const submit = async (): Promise<void> => {
    const text = request.trim();
    if (!text) return;
    setSubmitting(true);
    setError(null);
    try {
      const created =
        await CurateTasksService.createCurateTaskApiProjectsProjectIdExperimentsExperimentIdCurateTasksPost(
          projectId,
          experimentId,
          { request: text },
        );
      setTask(created);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">Curate</span>
        {task && <WorkbenchTag className="font-mono text-micro">{task.status}</WorkbenchTag>}
      </div>
      <Textarea
        value={request}
        onChange={(e) => setRequest(e.target.value)}
        placeholder='e.g. "Move run abc into baseline"'
        rows={2}
        disabled={inFlight || submitting}
      />
      <div className="flex items-center justify-end">
        <WorkbenchAction
          kind="primary"
          size="compact"
          className="h-7"
          disabled={!request.trim() || inFlight || submitting}
          onClick={() => void submit()}
        >
          {submitting || isRunning ? "…" : "Run"}
        </WorkbenchAction>
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {isWaitingApproval && taskId && (
        <ApprovalsInbox variant="detail" taskId={taskId} onDecided={() => undefined} />
      )}
    </div>
  );
}
