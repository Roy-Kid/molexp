import type { WorkspaceRunsResponse } from "./types";

const ENDPOINT = "/api/workspace/runs";

const handle = async <T>(response: Response, label: string): Promise<T> => {
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: string } | null) => body?.detail)
      .catch(() => null);
    throw new Error(`${label} failed (${response.status}): ${detail ?? response.statusText}`);
  }
  return response.json() as Promise<T>;
};

export interface ListRunsOptions {
  limit?: number;
}

export const workspaceRunsApi = {
  async listRuns(options: ListRunsOptions = {}): Promise<WorkspaceRunsResponse> {
    const params = new URLSearchParams();
    if (options.limit !== undefined) params.set("limit", String(options.limit));
    const url = params.size > 0 ? `${ENDPOINT}?${params.toString()}` : ENDPOINT;
    const response = await fetch(url);
    return handle<WorkspaceRunsResponse>(response, "List workspace runs");
  },
};
