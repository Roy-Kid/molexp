// ─────────────────────────────────────────────────────────────────────────────
// Breadcrumb trail — the ordered ancestor chain for the current selection,
// derived once from the entity graph. Replaces the ~140-line hand-rolled
// ``buildBreadcrumbs`` in useNavigationState; every trail link routes through
// the single ``entityPath`` URL builder. Lives in the shell, so EVERY view —
// including the dashboard landing pages that previously had no breadcrumb at
// all — gets a consistent trail.
// ─────────────────────────────────────────────────────────────────────────────

import { entityPath, SECTION_PATH } from "@/app/entities/paths";
import type { BreadcrumbItem, LeftPanelView, Selection, WorkspaceSnapshot } from "@/app/types";

const SECTION_ROOT: Record<LeftPanelView, BreadcrumbItem> = {
  projects: { label: "Experiments", to: SECTION_PATH.projects },
  workspace: { label: "Workspace", to: SECTION_PATH.workspace },
  runs: { label: "Runs", to: SECTION_PATH.runs },
  workflow: { label: "Workflows", to: SECTION_PATH.workflows },
  asset: { label: "Assets", to: SECTION_PATH.assets },
  library: { label: "Library", to: SECTION_PATH.library },
  agent: { label: "Agent Tasks", to: SECTION_PATH.agents },
  settings: { label: "Settings", to: SECTION_PATH.settings },
};

const crumb = (label: string, to?: string): BreadcrumbItem => (to ? { label, to } : { label });

export const buildTrail = (
  selection: Selection | null,
  leftPanelView: LeftPanelView,
  snapshot: WorkspaceSnapshot,
): BreadcrumbItem[] => {
  const root = SECTION_ROOT[leftPanelView];

  if (!selection) {
    // Section landing page — root only, not a link to itself.
    return [crumb(root.label)];
  }

  const link = (kind: Selection["objectType"], id: string, runId?: string): string | undefined =>
    entityPath({ kind, id, runId }, snapshot) ?? undefined;

  switch (selection.objectType) {
    case "project": {
      const project = snapshot.projects.find((p) => p.id === selection.objectId);
      return [root, crumb(project?.name ?? selection.objectId)];
    }
    case "experiment": {
      const experiment = snapshot.experiments.find((e) => e.id === selection.objectId);
      const project = experiment
        ? snapshot.projects.find((p) => p.id === experiment.projectId)
        : null;
      return [
        root,
        ...(project ? [crumb(project.name, link("project", project.id))] : []),
        crumb(experiment?.name ?? selection.objectId),
      ];
    }
    case "run": {
      const run = snapshot.runs.find((r) => r.id === selection.objectId);
      const experiment = run ? snapshot.experiments.find((e) => e.id === run.experimentId) : null;
      const project = run ? snapshot.projects.find((p) => p.id === run.projectId) : null;
      return [
        root,
        ...(project ? [crumb(project.name, link("project", project.id))] : []),
        ...(experiment ? [crumb(experiment.name, link("experiment", experiment.id))] : []),
        crumb(run?.name ?? selection.objectId),
      ];
    }
    case "task": {
      const run = snapshot.runs.find((r) => r.id === selection.runId);
      const experiment = run ? snapshot.experiments.find((e) => e.id === run.experimentId) : null;
      const project = run ? snapshot.projects.find((p) => p.id === run.projectId) : null;
      return [
        root,
        ...(project ? [crumb(project.name, link("project", project.id))] : []),
        ...(experiment ? [crumb(experiment.name, link("experiment", experiment.id))] : []),
        ...(run ? [crumb(run.name ?? run.id, link("run", run.id))] : []),
        crumb(selection.taskId),
      ];
    }
    case "workflow": {
      const workflow = snapshot.workflows.find((w) => w.id === selection.workflowId);
      return [root, crumb(workflow?.name ?? selection.workflowId)];
    }
    case "asset": {
      const asset = snapshot.assets.find((a) => a.id === selection.objectId);
      return [root, crumb(asset?.name ?? selection.objectId)];
    }
    case "agent": {
      if (selection.objectId === "new") return [root, crumb("New Task")];
      if (selection.objectId === "settings") return [root, crumb("Settings")];
      const session = snapshot.agentSessions.find((s) => s.id === selection.objectId);
      return [root, crumb(session?.goal ?? selection.objectId)];
    }
    case "workspace-file":
      return [root, crumb(selection.filePath.split("/").pop() ?? selection.filePath)];
  }
};
