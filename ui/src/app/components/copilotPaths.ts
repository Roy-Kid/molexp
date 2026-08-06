/**
 * Pure route mapping for Workspace Copilot next-actions (close-loop-04).
 */

import type { NextActionResponse } from "@/api/generated/models/NextActionResponse";
import { entityPath, runPath, SECTION_PATH } from "@/app/entities/paths";
import type { WorkspaceSnapshot } from "@/app/types";

/** Map a copilot next-action target to a navigable route. */
export const pathForNextAction = (
  action: NextActionResponse,
  snapshot: WorkspaceSnapshot,
): string | null => {
  const target = (action.target || "").trim();
  if (!target) return null;

  switch (action.kind) {
    case "diagnose_failed_run":
    case "retry_failed_run":
    case "review_stale_running": {
      const run = snapshot.runs.find((r) => r.id === target);
      if (run) return runPath(run.projectId, run.experimentId, run.id);
      return SECTION_PATH.runs;
    }
    case "answer_open_question": {
      const path = entityPath({ kind: "knowledge", id: target }, snapshot);
      return path ?? SECTION_PATH.knowledge;
    }
    case "review_orphan_artifact": {
      const path = entityPath({ kind: "asset", id: target }, snapshot);
      return path ?? SECTION_PATH.assets;
    }
    default: {
      const run = snapshot.runs.find((r) => r.id === target);
      if (run) return runPath(run.projectId, run.experimentId, run.id);
      return entityPath({ kind: "knowledge", id: target }, snapshot);
    }
  }
};
