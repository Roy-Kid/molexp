import type { MolqJobDetail, MolqJobsResponse, MolqTargetSummary } from "@/plugins/molq/types";

const BASE = "/api/plugins/molq";

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

export const molqApi = {
  async listTargets(): Promise<MolqTargetSummary[]> {
    const response = await fetch(`${BASE}/targets`);
    const body = await handle<{ targets: MolqTargetSummary[]; total: number }>(
      response,
      "List targets",
    );
    return body.targets;
  },

  async listJobs(target?: string, limit = 200): Promise<MolqJobsResponse> {
    const params = new URLSearchParams();
    if (target) params.set("target", target);
    params.set("limit", String(limit));
    const response = await fetch(`${BASE}/jobs?${params.toString()}`);
    return handle<MolqJobsResponse>(response, "List jobs");
  },

  async getJob(target: string, jobId: string): Promise<MolqJobDetail> {
    const params = new URLSearchParams({ target });
    const response = await fetch(`${BASE}/jobs/${encodeURIComponent(jobId)}?${params.toString()}`);
    return handle<MolqJobDetail>(response, "Get job");
  },

  streamLogs(target: string, jobId: string, stream: "stdout" | "stderr" = "stdout"): EventSource {
    const params = new URLSearchParams({ target, stream });
    return new EventSource(`${BASE}/jobs/${encodeURIComponent(jobId)}/logs?${params.toString()}`);
  },
};
