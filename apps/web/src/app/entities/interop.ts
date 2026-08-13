// Bridges the legacy ``Selection`` union to the unified ``EntityRef`` during the
// navigation rearchitecture. Once every surface speaks EntityRef this collapses
// away.

import type { EntityRef } from "@/app/entities/kinds";
import type { Selection } from "@/app/types";

export const refFromSelection = (selection: Selection): EntityRef => {
  if (selection.objectType === "task") {
    return { kind: "task", id: selection.taskId, runId: selection.runId };
  }
  if (selection.objectType === "workflow") {
    return { kind: "workflow", id: selection.workflowId };
  }
  return { kind: selection.objectType, id: selection.objectId };
};
