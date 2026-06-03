// ─────────────────────────────────────────────────────────────────────────────
// Relations — the typed edges of the workspace graph, declared once. Given any
// entity, ``resolveRelations`` returns its neighbours grouped by relationship.
// This single function powers the Related panel, the breadcrumb parent chain,
// and relation-aware command-palette results. Before this existed, every
// renderer hand-rolled its own ``setSelection`` jumps and most edges were
// simply missing (run→workflow, run→siblings, experiment→workflow, …).
//
// Only snapshot-derivable edges live here (synchronous, no network). Async
// edges that need a catalog/lineage fetch (asset→producer, task→produced
// assets) are resolved inside the owning viewer and are intentionally absent.
// ─────────────────────────────────────────────────────────────────────────────

import type { EntityRef } from "@/app/entities/kinds";
import type {
  ExperimentSummary,
  RunSummary,
  WorkflowSummary,
  WorkspaceSnapshot,
} from "@/app/types";

export interface RelationGroup {
  /** Stable machine key for the relationship. */
  relation: string;
  /** Display heading, e.g. "Experiment" or "Runs". */
  label: string;
  refs: EntityRef[];
}

const projectRef = (id: string, snapshot: WorkspaceSnapshot): EntityRef => {
  const project = snapshot.projects.find((p) => p.id === id);
  return { kind: "project", id, label: project?.name ?? id, status: project?.status };
};

const experimentRef = (experiment: ExperimentSummary): EntityRef => ({
  kind: "experiment",
  id: experiment.id,
  label: experiment.name,
  status: experiment.status,
});

const runRef = (run: RunSummary): EntityRef => ({
  kind: "run",
  id: run.id,
  label: run.name || run.id,
  status: run.status,
});

const workflowRef = (workflow: WorkflowSummary): EntityRef => ({
  kind: "workflow",
  id: workflow.id,
  label: workflow.name,
  status: workflow.status,
});

// The experiment→workflow edge: a workflow is bound to its experiment 1:1.
const workflowForExperiment = (
  experimentId: string,
  snapshot: WorkspaceSnapshot,
): WorkflowSummary | undefined => snapshot.workflows.find((w) => w.experimentId === experimentId);

const group = (relation: string, label: string, refs: EntityRef[]): RelationGroup => ({
  relation,
  label,
  refs,
});

const nonEmpty = (groups: RelationGroup[]): RelationGroup[] =>
  groups.filter((g) => g.refs.length > 0);

export const resolveRelations = (ref: EntityRef, snapshot: WorkspaceSnapshot): RelationGroup[] => {
  switch (ref.kind) {
    case "project": {
      const experiments = snapshot.experiments.filter((e) => e.projectId === ref.id);
      const workflows = snapshot.workflows.filter((w) => w.projectId === ref.id);
      const assets = snapshot.assets.filter((a) => a.projectId === ref.id);
      return nonEmpty([
        group("experiments", "Experiments", experiments.map(experimentRef)),
        group("workflows", "Workflows", workflows.map(workflowRef)),
        group(
          "assets",
          "Assets",
          assets.map((a) => ({ kind: "asset", id: a.id, label: a.name, status: a.status })),
        ),
      ]);
    }

    case "experiment": {
      const experiment = snapshot.experiments.find((e) => e.id === ref.id);
      if (!experiment) return [];
      const runs = snapshot.runs.filter((r) => r.experimentId === ref.id);
      const workflow = workflowForExperiment(ref.id, snapshot);
      return nonEmpty([
        group("project", "Project", [projectRef(experiment.projectId, snapshot)]),
        group("workflow", "Workflow", workflow ? [workflowRef(workflow)] : []),
        group("runs", "Runs", runs.map(runRef)),
      ]);
    }

    case "run": {
      const run = snapshot.runs.find((r) => r.id === ref.id);
      if (!run) return [];
      const experiment = snapshot.experiments.find((e) => e.id === run.experimentId);
      const workflow = workflowForExperiment(run.experimentId, snapshot);
      const siblings = snapshot.runs.filter(
        (r) => r.experimentId === run.experimentId && r.id !== run.id,
      );
      return nonEmpty([
        group("project", "Project", [projectRef(run.projectId, snapshot)]),
        group("experiment", "Experiment", experiment ? [experimentRef(experiment)] : []),
        group("workflow", "Workflow", workflow ? [workflowRef(workflow)] : []),
        group("siblings", "Sibling runs", siblings.map(runRef)),
      ]);
    }

    case "task": {
      const runId = ref.runId;
      const run = runId ? snapshot.runs.find((r) => r.id === runId) : undefined;
      if (!run) return [];
      const experiment = snapshot.experiments.find((e) => e.id === run.experimentId);
      return nonEmpty([
        group("run", "Run", [runRef(run)]),
        group("experiment", "Experiment", experiment ? [experimentRef(experiment)] : []),
      ]);
    }

    case "workflow": {
      const workflow = snapshot.workflows.find((w) => w.id === ref.id);
      if (!workflow) return [];
      const runs = snapshot.runs.filter((r) => r.experimentId === workflow.experimentId);
      const experiment = snapshot.experiments.find((e) => e.id === workflow.experimentId);
      return nonEmpty([
        group("project", "Project", [projectRef(workflow.projectId, snapshot)]),
        group("experiment", "Experiment", experiment ? [experimentRef(experiment)] : []),
        group("runs", "Runs", runs.map(runRef)),
      ]);
    }

    case "asset": {
      const asset = snapshot.assets.find((a) => a.id === ref.id);
      if (!asset?.projectId) return [];
      return nonEmpty([group("project", "Project", [projectRef(asset.projectId, snapshot)])]);
    }

    case "agent":
    case "workspace-file":
      // Agent sessions and raw files have no snapshot-level edges; their
      // viewers resolve producer/origin links asynchronously.
      return [];
  }
};
