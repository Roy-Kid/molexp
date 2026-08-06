/**
 * Shared POST .../analyze-failure client (close-loop-05).
 * Same service path as CLI/services — no client-side KnowledgeItem invent.
 */

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
  const res = await fetch(
    `/api/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(experimentId)}/runs/${encodeURIComponent(runId)}/analyze-failure`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        narrative: body.narrative ?? null,
        created_by: body.created_by ?? "ui",
        force: body.force ?? false,
      }),
    },
  );
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || res.statusText);
  }
  const json = (await res.json()) as { name?: string; path?: string };
  return {
    name: json.name?.trim() || "",
    path: json.path?.trim() || "",
  };
}
