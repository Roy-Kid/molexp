import { describe, expect, it } from "@rstest/core";

import { moveItemBefore, reconcileLayout } from "../useDashboardLayout";

describe("reconcileLayout", () => {
  it("returns defaults when nothing is stored", () => {
    expect(reconcileLayout(null, ["a", "b", "c"])).toEqual({
      order: ["a", "b", "c"],
      hidden: [],
    });
  });

  it("preserves stored order and merges new defaults at the end", () => {
    expect(
      reconcileLayout(
        { order: ["b", "a"], hidden: [] },
        ["a", "b", "c", "d"],
      ),
    ).toEqual({ order: ["b", "a", "c", "d"], hidden: [] });
  });

  it("drops stale ids that are no longer in defaults", () => {
    expect(
      reconcileLayout(
        { order: ["a", "gone", "b"], hidden: ["zombie"] },
        ["a", "b"],
      ),
    ).toEqual({ order: ["a", "b"], hidden: [] });
  });

  it("keeps hidden ids that still exist in defaults", () => {
    expect(
      reconcileLayout(
        { order: ["a", "b", "c"], hidden: ["b"] },
        ["a", "b", "c"],
      ),
    ).toEqual({ order: ["a", "b", "c"], hidden: ["b"] });
  });
});

describe("moveItemBefore (drop-on-target semantics)", () => {
  it("dragging upward drops the active item before the target", () => {
    expect(moveItemBefore(["a", "b", "c", "d"], "d", "a")).toEqual([
      "d",
      "a",
      "b",
      "c",
    ]);
  });

  it("dragging downward drops the active item after the target", () => {
    // "a" dropped on "c" lands at c's original index (2), which is after "c"
    // once "a" is removed — natural HTML5 drag feel.
    expect(moveItemBefore(["a", "b", "c", "d"], "a", "c")).toEqual([
      "b",
      "c",
      "a",
      "d",
    ]);
  });

  it("is a no-op when active equals over", () => {
    expect(moveItemBefore(["a", "b", "c"], "b", "b")).toEqual(["a", "b", "c"]);
  });

  it("returns the list unchanged when an id is missing", () => {
    expect(moveItemBefore(["a", "b"], "x", "a")).toEqual(["a", "b"]);
    expect(moveItemBefore(["a", "b"], "a", "x")).toEqual(["a", "b"]);
  });
});
