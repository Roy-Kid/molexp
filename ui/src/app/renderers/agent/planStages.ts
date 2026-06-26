// Canonical PlanMode stage list — the single source shared by the progress rail
// (left) and the deliverables panel (right). Mirrors the server's synthesized
// stage steps (`server/plan_runtime/record.py:_STAGE_LABELS`), keyed by the
// representative artifact kind each of the nine steps produces. `view` is the
// deliverable a stage renders; stages without one have no standalone document
// and render an empty panel when selected.

export type DeliverableView =
  | "report"
  | "spec"
  | "capabilities"
  | "topology"
  | "script"
  | "inputs"
  | "dryrun"
  | "review"
  | "execution";

export interface PlanStage {
  kind: string;
  label: string;
  view?: DeliverableView;
}

export const PLAN_STAGES: PlanStage[] = [
  { kind: "experiment_report", label: "Draft proposal", view: "report" },
  { kind: "experiment_spec", label: "Draft spec", view: "spec" },
  { kind: "capability_catalog", label: "Resolve capabilities", view: "capabilities" },
  { kind: "workflow_ir", label: "Workflow spec", view: "topology" },
  { kind: "workflow_source", label: "Tasks + tests", view: "script" },
  { kind: "input_set", label: "Input set", view: "inputs" },
  { kind: "execution_result", label: "Compile & dry-run", view: "dryrun" },
  { kind: "analysis_result", label: "Review", view: "review" },
  { kind: "execution_report", label: "Execution report", view: "execution" },
];

/** The stage selected by default when a plan session opens (the proposal). */
export const DEFAULT_PLAN_STAGE = "experiment_report";

export const planStage = (kind: string): PlanStage | undefined =>
  PLAN_STAGES.find((s) => s.kind === kind);
