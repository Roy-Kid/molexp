import { CheckCircle2, ShieldQuestion, XCircle } from "lucide-react";
import { type JSX, useCallback, useEffect, useState } from "react";
import type { PendingApprovalItem } from "@/api/generated/models/PendingApprovalItem";
import { ApprovalsService } from "@/api/generated/services/ApprovalsService";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { formatDateTime } from "@/lib/datetime";

/**
 * Approvals inbox — the ONE pending-approvals surface (vision-loop-01).
 *
 * Lists every request suspended plan/curate tasks are waiting on
 * (GET /api/approvals), lets the operator Approve / Reject (+ optional
 * reason), and refetches whenever the server pings the SSE stream
 * (/api/approvals/events) — a decision made anywhere (another tab, the CLI)
 * updates this view without polling.
 *
 * `taskId` narrows the inbox to one task's requests — the inline
 * "waiting for approval" banner inside a plan/curate task view reuses the
 * same cards and decision path as the hub-level inbox.
 */

const useApprovalsInbox = (): {
  items: PendingApprovalItem[];
  refetch: () => Promise<void>;
} => {
  const [items, setItems] = useState<PendingApprovalItem[]>([]);

  const refetch = useCallback(async () => {
    try {
      const response = await ApprovalsService.listPendingApprovalsApiApprovalsGet();
      setItems(response.items);
    } catch {
      // Inbox route unavailable (older server / transient) — show nothing
      // rather than an error wall on the Agents hub landing.
      setItems([]);
    }
  }, []);

  useEffect(() => {
    void refetch();
    const source = new EventSource("/api/approvals/events");
    const onChanged = (): void => {
      void refetch();
    };
    source.addEventListener("changed", onChanged);
    return () => {
      source.removeEventListener("changed", onChanged);
      source.close();
    };
  }, [refetch]);

  return { items, refetch };
};

const ApprovalCard = ({
  item,
  onDecided,
}: {
  item: PendingApprovalItem;
  onDecided: () => void;
}): JSX.Element => {
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const decide = async (granted: boolean): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      await ApprovalsService.decideApprovalApiApprovalsTaskKindTaskIdDecisionsPost(
        item.taskKind,
        item.taskId,
        { requestId: item.requestId, granted, reason: reason.trim() || undefined },
      );
      onDecided();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to record the decision.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-2 rounded-md border border-warning/40 bg-warning-soft/40 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <ShieldQuestion className="h-4 w-4 flex-none text-warning" />
        <span className="text-sm font-medium text-foreground">{item.intent}</span>
        <Badge variant="outline" className="px-1.5 py-0 text-[10px] font-medium">
          {item.taskKind}
        </Badge>
        <span className="ml-auto text-xs text-muted-foreground">
          {formatDateTime(item.requestedAt)}
        </span>
      </div>
      <p className="text-sm text-muted-foreground">{item.reason}</p>
      <p className="text-xs text-muted-foreground">
        run {item.runId} · {item.projectId}/{item.experimentId}
      </p>
      {rejecting && (
        <Textarea
          placeholder="Why is this rejected? (optional)"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          rows={2}
          disabled={busy}
        />
      )}
      {error && <p className="text-sm text-destructive">{error}</p>}
      <div className="flex items-center justify-end gap-2">
        {rejecting ? (
          <>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => setRejecting(false)}>
              Back
            </Button>
            <Button size="sm" variant="destructive" disabled={busy} onClick={() => decide(false)}>
              <XCircle className="h-3.5 w-3.5" />
              Confirm reject
            </Button>
          </>
        ) : (
          <>
            <Button size="sm" variant="outline" disabled={busy} onClick={() => setRejecting(true)}>
              <XCircle className="h-3.5 w-3.5" />
              Reject
            </Button>
            <Button size="sm" disabled={busy} onClick={() => decide(true)}>
              <CheckCircle2 className="h-3.5 w-3.5" />
              Approve
            </Button>
          </>
        )}
      </div>
    </div>
  );
};

export const ApprovalsInbox = ({
  taskId,
  onDecided,
}: {
  /** Narrow to one task's pending requests (inline task-view banner). */
  taskId?: string;
  /** Extra callback after a decision landed (e.g. resume status polling). */
  onDecided?: () => void;
}): JSX.Element | null => {
  const { items, refetch } = useApprovalsInbox();
  const visible = taskId ? items.filter((item) => item.taskId === taskId) : items;
  if (visible.length === 0) return null;

  const handleDecided = (): void => {
    void refetch();
    onDecided?.();
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 px-1">
        <p className="text-xs font-medium text-muted-foreground">
          {taskId ? "Waiting for your approval" : "Pending approvals"}
        </p>
        <Badge
          variant="outline"
          className="border-warning/25 bg-warning-soft px-1.5 py-0 text-[10px] font-medium text-warning-foreground"
        >
          {visible.length}
        </Badge>
      </div>
      {visible.map((item) => (
        <ApprovalCard
          key={`${item.taskId}:${item.requestId}`}
          item={item}
          onDecided={handleDecided}
        />
      ))}
    </div>
  );
};
