/**
 * Slim approve / reject / revise actions for a waiting plan — not a second
 * document card. The experiment plan book lives in the agent answer above.
 *
 * Approve = freeze + start realization (workflow codegen). There is no separate
 * "generate workflow" button; that is the approve path.
 */

import { CheckCircle2, Pencil, XCircle } from "lucide-react";
import { type JSX, useCallback, useEffect, useState } from "react";
import type { PendingApprovalItem } from "@/api/generated/models/PendingApprovalItem";
import { ApprovalsService } from "@/api/generated/services/ApprovalsService";
import { Textarea } from "@/components/ui/textarea";
import { WorkbenchAction, WorkbenchOperationState } from "@/components/workbench";

export const PlanDecisionBar = ({
  taskId,
  runId = null,
  onDecided,
}: {
  taskId: string;
  /** Match pending items by plan run when public task id diverges. */
  runId?: string | null;
  onDecided?: () => void;
}): JSX.Element | null => {
  const [item, setItem] = useState<PendingApprovalItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [comment, setComment] = useState("");
  const [rejecting, setRejecting] = useState(false);
  const [busy, setBusy] = useState<"approve" | "reject" | "revise" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await ApprovalsService.listPendingApprovalsApiApprovalsGet();
      const match =
        res.items.find(
          (i) =>
            i.taskKind === "plan" && (i.taskId === taskId || (runId != null && i.runId === runId)),
        ) ?? null;
      setItem(match);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load decisions.");
    } finally {
      setLoading(false);
    }
  }, [taskId, runId]);

  useEffect(() => {
    void load();
  }, [load]);

  const decide = async (action: "approve" | "reject" | "revise"): Promise<void> => {
    if (!item) return;
    setBusy(action);
    setError(null);
    try {
      const fieldValues: Record<string, unknown> = {};
      if (comment.trim()) fieldValues.operator_notes = comment.trim();
      await ApprovalsService.decideApprovalApiApprovalsTaskKindTaskIdDecisionsPost(
        item.taskKind,
        item.taskId,
        {
          requestId: item.requestId,
          action,
          reason: action === "reject" ? comment.trim() || undefined : undefined,
          fieldValues: Object.keys(fieldValues).length > 0 ? fieldValues : undefined,
        },
      );
      onDecided?.();
      setItem(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to record decision.");
    } finally {
      setBusy(null);
    }
  };

  if (loading) {
    return <WorkbenchOperationState kind="loading" density="compact" title="Loading decision…" />;
  }
  if (!item) {
    return (
      <div className="space-y-1 border-t border-border/60 pt-3 text-label text-muted-foreground">
        <p>
          No live approval request is attached to this task
          {runId ? ` (run ${runId})` : ""}. The plan worker may have restarted — open the Approvals
          bell, or re-run Plan Mode.
        </p>
        {error ? <WorkbenchOperationState kind="error" density="compact" title={error} /> : null}
      </div>
    );
  }

  return (
    <div className="space-y-2 border-t border-border/60 pt-3" aria-busy={busy !== null}>
      <p className="text-label text-muted-foreground">
        Approve freezes this plan and{" "}
        <strong className="text-foreground">starts workflow generation</strong> (codegen + compile).
        No separate continue step.
      </p>
      <Textarea
        placeholder="Comment (optional)…"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        rows={2}
        disabled={busy !== null}
        className="border-border/60 bg-muted/30 text-body-lg shadow-none"
      />
      {error ? <WorkbenchOperationState kind="error" density="compact" title={error} /> : null}
      {busy ? (
        <WorkbenchOperationState
          kind="running"
          density="compact"
          title={
            busy === "approve"
              ? "Approving — starting workflow generation…"
              : busy === "reject"
                ? "Rejecting…"
                : "Sending revise…"
          }
        />
      ) : null}
      <div className="flex items-center justify-end gap-2">
        {rejecting ? (
          <>
            <WorkbenchAction
              kind="ghost"
              size="compact"
              disabled={busy !== null}
              onClick={() => setRejecting(false)}
            >
              Back
            </WorkbenchAction>
            <WorkbenchAction
              kind="danger"
              size="compact"
              disabled={busy !== null}
              onClick={() => void decide("reject")}
            >
              <XCircle className="h-3.5 w-3.5" />
              Confirm reject
            </WorkbenchAction>
          </>
        ) : (
          <>
            <WorkbenchAction
              kind="ghost"
              size="compact"
              disabled={busy !== null}
              onClick={() => setRejecting(true)}
            >
              Reject
            </WorkbenchAction>
            <WorkbenchAction
              kind="secondary"
              size="compact"
              disabled={busy !== null}
              onClick={() => void decide("revise")}
            >
              <Pencil className="h-3.5 w-3.5" />
              Revise
            </WorkbenchAction>
            <WorkbenchAction
              kind="primary"
              size="compact"
              disabled={busy !== null}
              onClick={() => void decide("approve")}
            >
              <CheckCircle2 className="h-3.5 w-3.5" />
              Approve & generate workflow
            </WorkbenchAction>
          </>
        )}
      </div>
    </div>
  );
};
