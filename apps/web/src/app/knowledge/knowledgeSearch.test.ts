import { describe, expect, it } from "@rstest/core";

describe("selectionForSearchHit", () => {
  // Pure routing helper — imported directly (node env, no jsdom needed).
  it("routes entity hits to entity pages and notes to knowledge", async () => {
    const { selectionForSearchHit } = await import("./searchHitSelection");
    expect(
      selectionForSearchHit({
        path: "projects/p1/experiments/e1/runs/run-abc123",
        type: "workspace.run",
      }),
    ).toEqual({ objectType: "run", objectId: "abc123" });
    expect(
      selectionForSearchHit({ path: "projects/p1/experiments/e1", type: "workspace.experiment" }),
    ).toEqual({ objectType: "experiment", objectId: "e1" });
    expect(selectionForSearchHit({ path: "projects/p1", type: "workspace.project" })).toEqual({
      objectType: "project",
      objectId: "p1",
    });
    expect(selectionForSearchHit({ path: "protocols/gel-prep", type: "okf.note" })).toEqual({
      objectType: "knowledge",
      objectId: "protocols/gel-prep",
    });
    // An entity hit with an unparseable identity path degrades to knowledge,
    // never a broken entity link.
    expect(selectionForSearchHit({ path: "weird/place", type: "workspace.run" })).toEqual({
      objectType: "knowledge",
      objectId: "weird/place",
    });
  });
});
