// ─────────────────────────────────────────────────────────────────────────────
// The single place that knows how an entity maps to a URL. Everything that
// navigates — left nav, breadcrumb, related panel, command palette, in-page
// links — routes through ``entityPath`` so there is exactly one URL scheme to
// reason about. ``matchEntity`` is the inverse: URL → EntityRef.
// ─────────────────────────────────────────────────────────────────────────────

import type { EntityRef } from "@/app/entities/kinds";
import type { WorkspaceSnapshot } from "@/app/types";

const enc = encodeURIComponent;

/** Canonical run URL, built from ids the caller already holds. This is the one
 *  place the run route shape is defined — both ``entityPath`` (snapshot lookup)
 *  and callers that already have the parent ids (e.g. the runs dashboard, whose
 *  poller rows may include runs not in the view snapshot) route through it. */
export const runPath = (projectId: string, experimentId: string, runId: string): string =>
  `/projects/${enc(projectId)}/experiments/${enc(experimentId)}/runs/${enc(runId)}`;

/** Build the canonical URL for an entity ref, or ``null`` if it cannot be
 *  located in the current snapshot (e.g. a stale ref to a deleted run). */
export const entityPath = (ref: EntityRef, snapshot: WorkspaceSnapshot): string | null => {
  switch (ref.kind) {
    case "project":
      return `/projects/${enc(ref.id)}`;

    case "experiment": {
      const experiment = snapshot.experiments.find((e) => e.id === ref.id);
      if (!experiment) return null;
      return `/projects/${enc(experiment.projectId)}/experiments/${enc(experiment.id)}`;
    }

    case "run": {
      const run = snapshot.runs.find((r) => r.id === ref.id);
      if (!run) return null;
      return runPath(run.projectId, run.experimentId, run.id);
    }

    case "task": {
      const runId = ref.runId;
      if (!runId) return null;
      const run = snapshot.runs.find((r) => r.id === runId);
      if (!run) return null;
      return `${runPath(run.projectId, run.experimentId, run.id)}/tasks/${enc(ref.id)}`;
    }

    case "workflow":
      return `/workflows/${enc(ref.id)}`;

    case "asset":
      return `/assets/${enc(ref.id)}`;

    case "agent":
      return ref.id === "new" ? "/agent-tasks/new" : `/agent-tasks/${enc(ref.id)}`;

    case "knowledge":
      // id is a bundle-relative concept path (may contain "/"); bare id → browse.
      return ref.id ? `/knowledge/${ref.id.split("/").map(enc).join("/")}` : "/knowledge";

    case "workspace-file":
      // Files carry their kind in the query string; callers that have the file
      // kind should build the path via ``fileEntityPath`` instead.
      return `/workspace?file=${enc(ref.id)}`;
  }
};

/** Section landing routes — the collection pages reached from a breadcrumb
 *  root or the nav rail. */
export const SECTION_PATH = {
  projects: "/projects",
  workspace: "/workspace",
  runs: "/runs",
  workflows: "/workflows",
  assets: "/assets",
  agents: "/agent-tasks",
  knowledge: "/knowledge",
  settings: "/settings",
} as const;
