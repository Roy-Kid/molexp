/**
 * Global pending-approvals badge (ui-approvals-bell-01).
 * Reuses GET /api/approvals + SSE; opens the same ApprovalsInbox drawer.
 */

import { Bell } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { ApprovalsService } from "@/api/generated/services/ApprovalsService";
import { ApprovalsInbox } from "@/app/renderers/agent/ApprovalsInbox";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

export function ApprovalsBell(): JSX.Element {
  const [count, setCount] = useState(0);
  const [open, setOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const response = await ApprovalsService.listPendingApprovalsApiApprovalsGet();
      setCount(response.items?.length ?? 0);
    } catch {
      // Non-fatal — badge stays at last known count.
    }
  }, []);

  useEffect(() => {
    void refresh();
    const source = new EventSource("/api/approvals/events");
    source.onmessage = () => {
      void refresh();
    };
    source.onerror = () => {
      /* browser will reconnect EventSource */
    };
    return () => source.close();
  }, [refresh]);

  return (
    <>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="relative h-8 w-8 flex-none"
              onClick={() => setOpen(true)}
              aria-label={
                count > 0 ? `${count} pending approvals` : "Approvals inbox"
              }
            >
              <Bell className="h-4 w-4" />
              {count > 0 && (
                <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-destructive px-1 text-[10px] font-medium text-destructive-foreground">
                  {count > 99 ? "99+" : count}
                </span>
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            {count > 0 ? `${count} pending approval(s)` : "Approvals"}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent className="w-full sm:max-w-md">
          <SheetHeader>
            <SheetTitle>Approvals</SheetTitle>
            <SheetDescription className="sr-only">Pending decisions</SheetDescription>
          </SheetHeader>
          <div className="mt-4">
            <ApprovalsInbox
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
