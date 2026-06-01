/**
 * Tests for workspace_targets MSW handlers — the four
 * /api/workspace/targets endpoints used by RemoteWorkspacesPanel.
 *
 * Strategy: structural method × path coverage from the exported
 * handler list. Behaviour (status codes, response shapes) is exercised
 * by the live dev:mock UI flow; the current rstest+msw test infra
 * doesn't bring in jsdom/fetch-interception, so resolver-level tests
 * would require a heavier setup than this spec warrants.
 */

import { describe, expect, it } from "@rstest/core";

import { workspaceTargetsHandlers } from "../workspace_targets";

interface HandlerInfo {
  method: string;
  path: string;
}

const handlerSummary = (): HandlerInfo[] =>
  workspaceTargetsHandlers.map((h) => {
    const info = (h as unknown as { info: HandlerInfo }).info;
    return { method: info.method, path: info.path };
  });

describe("workspace_targets handlers — coverage matrix", () => {
  it("exports handlers for the four workspace-targets endpoints in spec order", () => {
    expect(handlerSummary()).toEqual([
      { method: "GET", path: "/api/workspace/targets" },
      { method: "POST", path: "/api/workspace/targets" },
      { method: "DELETE", path: "/api/workspace/targets/:name" },
      { method: "POST", path: "/api/workspace/targets/:name/test" },
    ]);
  });

  it("contains a GET list endpoint", () => {
    expect(handlerSummary()).toContainEqual({
      method: "GET",
      path: "/api/workspace/targets",
    });
  });

  it("contains a POST create endpoint", () => {
    expect(handlerSummary()).toContainEqual({
      method: "POST",
      path: "/api/workspace/targets",
    });
  });

  it("contains a DELETE endpoint with a :name parameter", () => {
    expect(handlerSummary()).toContainEqual({
      method: "DELETE",
      path: "/api/workspace/targets/:name",
    });
  });

  it("contains a POST test endpoint with a :name parameter", () => {
    expect(handlerSummary()).toContainEqual({
      method: "POST",
      path: "/api/workspace/targets/:name/test",
    });
  });
});
