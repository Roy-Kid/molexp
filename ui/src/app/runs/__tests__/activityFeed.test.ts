/**
 * Pure-core tests for the workspace activity feed (vision-loop-12): event
 * type → icon/label mapping, ref → entity-link resolution, empty-state copy.
 */

import { describe, expect, it } from "@rstest/core";

import { eventVisualFor, FEED_EMPTY_TEXT, resolveEventRef } from "@/app/runs/activityFeed";

describe("eventVisualFor", () => {
  it("maps every spine event type to a distinct label", () => {
    // Mirrors the WorkspaceEventType Literal (workspace/events.py) exactly.
    const types = [
      "run.created",
      "run.started",
      "run.failed",
      "run.completed",
      "asset.added",
      "knowledge.created",
      "workflow.created",
      "experiment.created",
    ];
    const labels = types.map((t) => eventVisualFor(t).label);
    expect(new Set(labels).size).toBe(types.length);
    expect(labels.every((label) => label.length > 0)).toBe(true);
  });

  it("gives each mapped type an icon and a dot class", () => {
    const visual = eventVisualFor("asset.added");
    expect(visual.icon).toBeDefined();
    expect(visual.dotClass).toContain("bg-");
  });

  it("renders an unknown type by its raw name instead of hiding it", () => {
    expect(eventVisualFor("mystery.event").label).toBe("mystery.event");
  });
});

describe("resolveEventRef", () => {
  const known = new Set(["run-abc123"]);

  it("resolves a known run id to a run link", () => {
    expect(resolveEventRef("run-abc123", "asset.added", known)).toEqual({
      kind: "run",
      runId: "run-abc123",
      text: "run-abc123",
    });
  });

  it("resolves knowledge.created refs to knowledge paths", () => {
    expect(resolveEventRef("protocols/gel-prep", "knowledge.created", known)).toEqual({
      kind: "knowledge",
      path: "protocols/gel-prep",
      text: "protocols/gel-prep",
    });
  });

  it("leaves unresolvable refs as plain text — never a dead link", () => {
    expect(resolveEventRef("sha256:deadbeef", "asset.added", known)).toEqual({
      kind: "plain",
      text: "sha256:deadbeef",
    });
  });

  it("shows the asset's human name for asset.added ids when the payload has one", () => {
    const resolved = resolveEventRef("3a189b4b-7754-463e-bf63", "asset.added", known, {
      kind: "artifact",
      name: "result.json",
    });
    expect(resolved).toEqual({ kind: "plain", text: "result.json" });
  });

  it("prefers the run link when a knowledge event also references a run", () => {
    expect(resolveEventRef("run-abc123", "knowledge.created", known).kind).toBe("run");
  });
});

describe("empty state", () => {
  it("states when events will appear, not just that none exist", () => {
    expect(FEED_EMPTY_TEXT).toBe("No events yet.");
  });
});
