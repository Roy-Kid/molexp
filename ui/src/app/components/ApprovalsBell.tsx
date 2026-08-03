/**
 * Global pending-approvals badge — list-only drawer.
 * Full review happens on the agent task surface (not duplicated here).
 */

import { Bell, Loader2 } from "lucide-react";
import { type JSX, useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { PendingApprovalItem } from "@/api/generated/models/PendingApprovalItem";
import { ApprovalsService } from "@/api/generated/services/ApprovalsService";
import { ApprovalsInbox } from "@/app/renderers/agent/ApprovalsInbox";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  WorkbenchAction,
  WorkbenchIconAction,
  WorkbenchOperationState,
} from "@/components/workbench";
import { cn } from "@/lib/utils";

export function ApprovalsBell(): JSX.Element {
  const navigate = useNavigate();
  const [count, setCount] = useState(0);
  const [countLoading, setCountLoading] = useState(true);
  const [countError, setCountError] = useState<string | null>(null);
  const [countAnnouncement, setCountAnnouncement] = useState<string | null>(null);
  const [streamError, setStreamError] = useState(false);
  const [open, setOpen] = useState(false);
  const countRef = useRef(0);

  const refresh = useCallback(async () => {
    setCountLoading(true);
    try {
      const response = await ApprovalsService.listPendingApprovalsApiApprovalsGet();
      const nextCount = response.items?.length ?? 0;
      if (nextCount !== countRef.current) {
        setCountAnnouncement(
          `Approval count updated: ${nextCount} pending approval${nextCount === 1 ? "" : "s"}.`,
        );
        countRef.current = nextCount;
      }
      setCount(nextCount);
      setCountError(null);
    } catch (err) {
      setCountError(err instanceof Error ? err.message : "Failed to load approval count.");
    } finally {
      setCountLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const source = new EventSource("/api/approvals/events");
    source.onmessage = () => {
      void refresh();
    };
    source.onopen = () => {
      setStreamError(false);
    };
    source.onerror = () => {
      setStreamError(true);
    };
    return () => source.close();
  }, [refresh]);

  useEffect(() => {
    if (!countAnnouncement) return;
    const handle = window.setTimeout(() => setCountAnnouncement(null), 3000);
    return () => window.clearTimeout(handle);
  }, [countAnnouncement]);

  const openItem = useCallback(
    (item: PendingApprovalItem) => {
      setOpen(false);
      // taskId is the single public conversation id (agent-task / plan session).
      navigate(`/agent-tasks/${encodeURIComponent(item.taskId)}`);
    },
    [navigate],
  );

  const tooltipLabel = countLoading
    ? "Loading approvals…"
    : countError
      ? "Approval count unavailable"
      : count > 0
        ? `${count} pending approval(s)`
        : "Approvals";

  return (
    <>
      {countLoading && (
        <WorkbenchOperationState
          kind="loading"
          density="inline"
          title="Loading approval count…"
          className="sr-only"
        />
      )}
      {countAnnouncement && (
        <WorkbenchOperationState
          kind="success"
          density="inline"
          title={countAnnouncement}
          className="sr-only"
        />
      )}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <WorkbenchIconAction
              label={tooltipLabel}
              kind="ghost"
              size="default"
              className={cn("relative flex-none", countError && "text-status-failed-foreground")}
              onClick={() => setOpen(true)}
              aria-busy={countLoading}
            >
              {countLoading ? (
                <Loader2
                  className="mol-motion-progress-spin h-4 w-4 text-status-running"
                  aria-hidden
                />
              ) : (
                <Bell className="h-4 w-4" />
              )}
              {count > 0 && (
                <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-info px-1 text-micro font-medium text-info-foreground">
                  {count > 99 ? "99+" : count}
                </span>
              )}
              {countError && count === 0 && (
                <span
                  className="absolute right-0 top-0 h-1.5 w-1.5 rounded-full bg-status-failed"
                  aria-hidden
                />
              )}
            </WorkbenchIconAction>
          </TooltipTrigger>
          <TooltipContent side="bottom">{tooltipLabel}</TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent className="flex w-full flex-col sm:max-w-sm">
          <SheetHeader>
            <SheetTitle>Approvals</SheetTitle>
            <SheetDescription>Open a task to review and decide.</SheetDescription>
          </SheetHeader>
          <div className="mt-4 min-h-0 flex-1 space-y-3 overflow-y-auto">
            {countError && (
              <WorkbenchOperationState
                kind="error"
                density="compact"
                title="Unavailable"
                detail={countError}
                action={
                  <WorkbenchAction kind="secondary" size="compact" onClick={() => void refresh()}>
                    Retry
                  </WorkbenchAction>
                }
              />
            )}
            {streamError && (
              <WorkbenchOperationState kind="running" density="compact" title="Reconnecting…" />
            )}
            <ApprovalsInbox
              variant="list"
              showStreamStatus={false}
              onOpenItem={openItem}
              onDecided={() => {
                void refresh();
              }}
            />
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}
