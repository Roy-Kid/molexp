/**
 * RED test for wiring the shared RunMetricsView into RunInspector's
 * Metrics tab (ac-004).
 *
 * Rstest runs in a node environment without jsdom (see rstest.config.ts:
 * testEnvironment: "node"), so we cannot render the component. Following
 * the repo convention (app/settings/__tests__/AddRemoteWorkspaceForm.test.tsx)
 * we parse the component source and assert the wiring contract:
 *
 *   - the Metrics TabsContent renders <RunMetricsView ...>, and
 *   - the old "Metrics view not wired yet" placeholder string is gone.
 *
 * This is RED today: RunInspector.tsx still renders the placeholder.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SOURCE_PATH = resolve(__dirname, "./RunInspector.tsx");
const source = readFileSync(SOURCE_PATH, "utf8");

describe("RunInspector metrics tab — RunMetricsView wiring (ac-004)", () => {
  it("imports and renders RunMetricsView", () => {
    expect(source).toContain("RunMetricsView");
    expect(source).toContain("<RunMetricsView");
  });

  it("passes the run's coordinate props into RunMetricsView", () => {
    expect(source).toMatch(/<RunMetricsView[\s\S]*projectId[\s\S]*experimentId[\s\S]*runId/);
  });

  it("no longer renders the 'Metrics view not wired yet' placeholder", () => {
    expect(source).not.toContain("Metrics view not wired yet");
  });
});
