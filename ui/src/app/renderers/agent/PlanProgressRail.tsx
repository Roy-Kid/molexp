import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";
import type { JSX } from "react";
import { completedStageKinds } from "@/app/renderers/agentEvents";
import type { ApiSessionEvent } from "@/app/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { PLAN_STAGES } from "./planStages";

// ---------------------------------------------------------------------------
// Vertical PlanMode progress rail — breadcrumb + navigator for Deliverables.
//
// Completion sources (merged):
//   1. `tool_call_completed` events with `payload.result.artifact`
//   2. live `artifactKinds` from GET /plans/{runId} (disk truth while SSE lags)
// ---------------------------------------------------------------------------

type StageState = "done" | "current" | "failed" | "pending";

const StepNode = ({ state }: { state: StageState }): JSX.Element => {
  if (state === "done") return <CheckCircle2 className="h-4 w-4 text-success" />;
  if (state === "current")
    return <Loader2 className="h-4 w-4 mol-motion-progress-spin text-info" />;
  if (state === "failed") return <XCircle className="h-4 w-4 text-destructive" />;
  return <Circle className="h-4 w-4 text-muted-foreground/40" />;
};

export const PlanProgressRail = ({
  events,
  status,
  selectedKind,
  onSelectStage,
  artifactKinds = [],
}: {
  events: ApiSessionEvent[];
  status: string;
  selectedKind: string;
  onSelectStage: (kind: string) => void;
  /** Authoritative kinds from GET /plans — keeps the rail honest while events lag. */
  artifactKinds?: readonly string[];
}): JSX.Element => {
  const fromEvents = completedStageKinds(events);
  const completed = new Set<string>([...fromEvents, ...artifactKinds]);
  // The --execute tail is opt-in: its stages appear only when the plan
  // actually produced their artifacts (a nine-step plan shows nine steps).
  const stages = PLAN_STAGES.filter((s) => !s.executeTail || completed.has(s.kind));
  const lastDone = stages.reduce((acc, s, i) => (completed.has(s.kind) ? i : acc), -1);
  const succeeded = status === "succeeded" || status === "completed";
  const failed = status === "failed" || status === "cancelled";
  const running = status === "running" || status === "waiting_approval";

  const stateOf = (i: number): StageState => {
    if (completed.has(stages[i]?.kind ?? "")) return "done";
    if (succeeded) return "done";
    if (i === lastDone + 1) {
      if (failed) return "failed";
      if (running || !succeeded) return "current";
    }
    if (i < lastDone) return "done";
    return "pending";
  };

  return (
    <div className="flex h-full w-[210px] flex-none flex-col border-r border-border/60 bg-background">
      <div className="flex-none px-4 py-3 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        PlanMode progress
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <ol className="relative px-4 pb-4">
          <div className="absolute bottom-5 left-[30px] top-4 w-px bg-border/70" aria-hidden />
          {stages.map((stage, i) => {
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
