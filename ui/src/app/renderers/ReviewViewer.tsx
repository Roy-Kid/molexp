import { Check, ClipboardCheck, ExternalLink, X } from "lucide-react";
import type { JSX } from "react";
import { useMemo, useState } from "react";
import { EntityHeader } from "@/app/components/entity/EntityPage";
import { StatusBadge } from "@/app/components/entity/StatusBadge";
import { reviewsApi } from "@/app/state/api";
import { useNavigationState } from "@/app/state/useNavigationState";
import type { RendererProps } from "@/app/types";
import { Button } from "@/components/ui/button";

export const ReviewViewer = ({ selection, snapshot, onRefresh }: RendererProps): JSX.Element => {
  const nav = useNavigationState(snapshot);
  const review =
    selection.objectType === "review"
      ? snapshot.reviews.find((item) => item.id === selection.objectId)
      : null;
  const [busy, setBusy] = useState<"approve" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canResolve = review?.status === "pending";
  const relatedTask = useMemo(
    () => snapshot.agentSessions.find((task) => task.id === review?.taskId),
    [review?.taskId, snapshot.agentSessions],
  );

  const resolve = async (decision: "approve" | "reject"): Promise<void> => {
    if (!review) return;
    setBusy(decision);
    setError(null);
    try {
      if (decision === "approve") {
        await reviewsApi.approve(review.id);
      } else {
        await reviewsApi.reject(review.id);
      }
      onRefresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(null);
    }
  };

  if (!review) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-muted-foreground">
        Review not found.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-background">
      <EntityHeader
        breadcrumbs={nav.breadcrumbs}
        canNavigateUp={nav.canNavigateUp}
        onNavigateUp={nav.navigateUp}
        icon={ClipboardCheck}
        title={review.title}
        status={review.status}
      />

      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 px-4 py-4 md:px-8">
        <div className="flex items-center gap-2">
          <StatusBadge status={review.kind} />
          <StatusBadge status={review.riskLevel} />
        </div>

        {review.description ? (
          <p className="text-sm text-muted-foreground">{review.description}</p>
        ) : null}

        <dl className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-2 text-sm">
          <dt className="text-muted-foreground">Review ID</dt>
          <dd className="font-mono text-xs">{review.id}</dd>
          <dt className="text-muted-foreground">Created</dt>
          <dd>{new Date(review.createdAt).toLocaleString()}</dd>
          <dt className="text-muted-foreground">Task</dt>
          <dd>
            {review.taskId ? (
              <Button
                variant="link"
                size="sm"
                className="h-auto p-0 text-sm"
                onClick={() =>
                  nav.setSelection({ objectType: "agent", objectId: review.taskId ?? "" })
                }
              >
                <ExternalLink className="mr-1 h-3.5 w-3.5" />
                {relatedTask?.goalDescription ?? review.taskId}
              </Button>
            ) : (
              "None"
            )}
          </dd>
        </dl>

        {error ? <p className="text-sm text-destructive">{error}</p> : null}

        {canResolve ? (
          <div className="flex gap-2 pt-2">
            <Button disabled={busy !== null} onClick={() => void resolve("approve")}>
              <Check className="mr-1 h-4 w-4" />
              Approve
            </Button>
            <Button
              variant="outline"
              disabled={busy !== null}
              onClick={() => void resolve("reject")}
            >
              <X className="mr-1 h-4 w-4" />
              Reject
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  );
};
