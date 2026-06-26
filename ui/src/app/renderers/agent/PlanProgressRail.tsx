import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";
import type { JSX } from "react";
import { completedStageKinds } from "@/app/renderers/agentEvents";
import type { ApiSessionEvent } from "@/app/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { PLAN_STAGES } from "./planStages";

// ---------------------------------------------------------------------------
// Vertical PlanMode progress rail — a breadcrumb down the left edge of a plan
// session that shows how far the pipeline ran AND navigates the right panel:
// clicking a stage selects it (single highlight) and the Deliverables panel
// shows that stage's document, or an empty panel when the stage produced none.
// Stage list comes from the shared `planStages` source.
// ---------------------------------------------------------------------------

type StageState = "done" | "current" | "failed" | "pending";

const StepNode = ({ state }: { state: StageState }): JSX.Element => {
  if (state === "done") return <CheckCircle2 className="h-4 w-4 text-success" />;
  if (state === "current") return <Loader2 className="h-4 w-4 animate-spin text-info" />;
  if (state === "failed") return <XCircle className="h-4 w-4 text-destructive" />;
  return <Circle className="h-4 w-4 text-muted-foreground/40" />;
};

export const PlanProgressRail = ({
  events,
  status,
  selectedKind,
  onSelectStage,
}: {
  events: ApiSessionEvent[];
  status: string;
  selectedKind: string;
  onSelectStage: (kind: string) => void;
}): JSX.Element => {
  const completed = completedStageKinds(events);
  const lastDone = PLAN_STAGES.reduce((acc, s, i) => (completed.has(s.kind) ? i : acc), -1);
  const succeeded = status === "succeeded" || status === "completed";
  const failed = status === "failed" || status === "cancelled";

  const stateOf = (i: number): StageState => {
    if (succeeded || i <= lastDone) return "done";
    if (i === lastDone + 1) return failed ? "failed" : "current";
    return "pending";
  };

  return (
    <div className="flex h-full w-[210px] flex-none flex-col border-r border-border/60 bg-background">
      <div className="flex-none px-4 py-3 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        PlanMode progress
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <ol className="relative px-4 pb-4">
          {/* connector spine, behind the nodes (aligned to node centers) */}
          <div className="absolute bottom-5 left-[30px] top-4 w-px bg-border/70" aria-hidden />
          {PLAN_STAGES.map((stage, i) => {
            const state = stateOf(i);
            const selected = stage.kind === selectedKind;
            return (
              <li key={stage.kind} className="relative py-px">
                <button
                  type="button"
                  onClick={() => onSelectStage(stage.kind)}
                  aria-current={selected}
                  className={cn(
                    "flex w-full items-start gap-2.5 rounded-md px-1 py-1 text-left transition-colors",
                    selected ? "bg-primary/10" : "hover:bg-muted/60",
                  )}
                >
                  <span className="relative z-10 flex h-5 w-5 flex-none items-center justify-center rounded-full bg-background">
                    <StepNode state={state} />
                  </span>
                  <span
                    className={cn(
                      "pt-0.5 text-xs leading-tight",
                      selected
                        ? "font-semibold text-primary"
                        : state === "pending"
                          ? "text-muted-foreground/50"
                          : state === "current"
                            ? "font-medium text-foreground"
                            : state === "failed"
                              ? "text-destructive"
                              : "text-foreground/80",
                    )}
                  >
                    {stage.label}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </ScrollArea>
    </div>
  );
};
