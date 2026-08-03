// Canonical PlanOrchestrator stage list — shared by the progress rail (left)
// and the deliverables panel (right). Mirrors services/plan_runtime/record.py
// `_STAGE_LABELS`, keyed by the representative artifact kind each stage produces.

export type DeliverableView =
  | "board"
  | "review"
  | "frozen"
  | "report"
  | "spec"
  | "bound"
  | "script"
  | "tests"
  | "dryrun"
  | "intervention"
  | "final"
  | "audit"
  /** Legacy / secondary views still renderable from PlanDetailResponse fields. */
  | "capabilities"
  | "topology"
  | "inputs"
  | "execution";

export interface PlanStage {
  kind: string;
  label: string;
  view?: DeliverableView;
  /** True for the opt-in execute tail — rail hides unless artifacts exist. */
  executeTail?: boolean;
}

export const PLAN_STAGES: PlanStage[] = [
  { kind: "experiment_plan", label: "Task board", view: "board" },
  { kind: "review_pack", label: "Review gate", view: "review" },
  { kind: "analysis_result", label: "Review decision", view: "review" },
  { kind: "frozen_experiment_plan", label: "Frozen plan", view: "frozen" },
  { kind: "plan_report", label: "Plan report", view: "report" },
  { kind: "experiment_spec", label: "Experiment spec", view: "spec" },
  { kind: "bound_workflow", label: "Bound tasks", view: "bound" },
  { kind: "workflow_source", label: "Workflow source", view: "script" },
  { kind: "test_source", label: "Per-task tests", view: "tests" },
  { kind: "execution_result", label: "Compile", view: "dryrun" },
  { kind: "intervention_request", label: "Intervention", view: "intervention" },
  { kind: "final_report", label: "Run & final report", view: "final", executeTail: true },
  { kind: "audit_report", label: "Audit trail", view: "audit", executeTail: true },
];

/** Default stage when a plan session opens. */
export const DEFAULT_PLAN_STAGE = "experiment_plan";

export const planStage = (kind: string): PlanStage | undefined =>
  PLAN_STAGES.find((s) => s.kind === kind);
