/**
 * Behavioral tests for the POST /api/workspace/open mock —
 * create_if_missing semantics (ui-creation-entries).
 *
 * Resolves requests against the handler list via msw's getResponse(), so
 * no fetch interception or jsdom is needed in the node test env. The
 * known-paths set is module-level state; every test uses a unique path.
 */

import { describe, expect, it } from "@rstest/core";
import { getResponse } from "msw";

import { workspaceHandlers } from "../workspace";

// msw resolves the handlers' relative paths against location, which the
// node test env lacks — pin it before any matching happens.
Reflect.set(globalThis, "location", new URL("http://localhost/"));

const openRequest = (body: Record<string, unknown>): Request =>
  new Request("http://localhost/api/workspace/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

describe("POST /api/workspace/open — create_if_missing semantics", () => {
  it("404s an unknown path when create_if_missing is omitted", async () => {
    const response = await getResponse(workspaceHandlers, openRequest({ path: "/unknown/omitted" }));
    expect(response?.status).toBe(404);
    const body = (await response?.json()) as { detail?: string };
    expect(body.detail).toBe("Workspace path not found");
  });

  it("404s an unknown path when create_if_missing is false", async () => {
    const response = await getResponse(
      workspaceHandlers,
      openRequest({ path: "/unknown/explicit-false", create_if_missing: false }),
    );
    expect(response?.status).toBe(404);
  });

  it("creates and opens an unknown path when create_if_missing is true", async () => {
    const response = await getResponse(
      workspaceHandlers,
      openRequest({ path: "/unknown/created", create_if_missing: true }),
    );
    expect(response?.status).toBe(200);
    const body = (await response?.json()) as { root?: string };
    expect(body.root).toBe("/unknown/created");
  });

  it("persists a created path — reopen without the flag succeeds", async () => {
    await getResponse(
      workspaceHandlers,
      openRequest({ path: "/unknown/persisted", create_if_missing: true }),
    );
    const reopen = await getResponse(workspaceHandlers, openRequest({ path: "/unknown/persisted" }));
    expect(reopen?.status).toBe(200);
  });

  it("opens the default mock root without any flag", async () => {
    const response = await getResponse(workspaceHandlers, openRequest({ path: "/mock-workspace" }));
    expect(response?.status).toBe(200);
  });
});
