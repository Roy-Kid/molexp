/**
 * Shared analyze-failure client (close-loop-05).
 * Uses the generated RunsService — same path as CLI/services (Python ≡ UI).
 */

import { RunsService } from "@/api/generated/services/RunsService";

export interface AnalyzeFailureResult {
  name: string;
  path: string;
}

export async function postAnalyzeFailure(
  projectId: string,
  experimentId: string,
  runId: string,
  body: { narrative?: string; created_by?: string; force?: boolean } = {},
): Promise<AnalyzeFailureResult> {
  const res =
    await RunsService.analyzeRunFailureRouteApiProjectsProjectIdExperimentsExperimentIdRunsRunIdAnalyzeFailurePost(
      projectId,
      experimentId,
      runId,
      {
        narrative: body.narrative ?? null,
        created_by: body.created_by ?? "ui",
        force: body.force ?? false,
      },
    );
  // Generated client types this as any/object; normalize to paths the UI expects.
  const json = res as { name?: string; path?: string };
  return {
    name: (json.name ?? "").trim(),
    path: (json.path ?? "").trim(),
  };
}
