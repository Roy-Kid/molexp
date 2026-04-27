/**
 * Tests for the workspace runs API client. Filtering moved entirely to the
 * client layer (see aggregates.ts), so the API now only knows about a
 * pagination limit; this suite verifies URL composition and error format.
 */

import { afterEach, describe, expect, it, rs } from "@rstest/core";

import { workspaceRunsApi } from "@/app/runs/api";

afterEach(() => {
  rs.restoreAllMocks();
});

const okJson = (body: object): Response =>
  new Response(JSON.stringify(body), { status: 200 });

const emptyResponse = (): Response =>
  okJson({
    runs: [],
    stats: { total: 0, running: 0, pending: 0, failed: 0, succeeded: 0 },
    total: 0,
    truncated: false,
  });

describe("workspaceRunsApi.listRuns", () => {
  it("calls the bare endpoint when no options are provided", async () => {
    const fetchSpy = rs.spyOn(globalThis, "fetch").mockResolvedValue(emptyResponse());

    await workspaceRunsApi.listRuns();

    expect(fetchSpy).toHaveBeenCalledWith("/api/workspace/runs");
  });

  it("encodes a limit when provided", async () => {
    const fetchSpy = rs.spyOn(globalThis, "fetch").mockResolvedValue(emptyResponse());

    await workspaceRunsApi.listRuns({ limit: 500 });

    const url = (fetchSpy.mock.calls[0]?.[0] as string) ?? "";
    expect(url).toBe("/api/workspace/runs?limit=500");
  });

  it("throws a descriptive error on non-OK response", async () => {
    rs.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "boom" }), { status: 500 }),
    );

    await expect(workspaceRunsApi.listRuns()).rejects.toThrow(
      /List workspace runs failed \(500\): boom/,
    );
  });
});
