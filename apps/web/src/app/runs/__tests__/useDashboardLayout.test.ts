import { describe, expect, it } from "@rstest/core";

import { applyReorder, reconcileLayout } from "../useDashboardLayout";

const r = (id: string, panels: string[]) => ({ id, panels });

describe("reconcileLayout", () => {
  it("returns a row per default when nothing is stored", () => {
    const next = reconcileLayout(null, ["a", "b", "c"]);
    expect(next.hidden).toEqual([]);
    expect(next.rows.map((row) => row.panels)).toEqual([["a"], ["b"], ["c"]]);
    expect(new Set(next.rows.map((row) => row.id)).size).toBe(3);
  });

  it("preserves stored row composition and order", () => {
    const stored = {
      rows: [r("row-1", ["b", "a"]), r("row-2", ["c"])],
      hidden: [],
    };
    expect(reconcileLayout(stored, ["a", "b", "c"])).toEqual(stored);
  });

  it("appends missing visible defaults as new single-panel rows", () => {
    const stored = { rows: [r("row-1", ["a", "b"])], hidden: [] };
    const next = reconcileLayout(stored, ["a", "b", "c", "d"]);
    expect(next.rows.length).toBe(3);
    expect(next.rows[0]).toEqual(r("row-1", ["a", "b"]));
    expect(next.rows[1].panels).toEqual(["c"]);
    expect(next.rows[2].panels).toEqual(["d"]);
  });

  it("does not re-add hidden defaults", () => {
    const stored = { rows: [r("row-1", ["a"])], hidden: ["b"] };
    const next = reconcileLayout(stored, ["a", "b", "c"]);
    expect(next.hidden).toEqual(["b"]);
    expect(next.rows.flatMap((row) => row.panels)).toEqual(["a", "c"]);
  });

  it("drops stale ids that are no longer in defaults", () => {
    const stored = {
      rows: [r("row-1", ["a", "ghost"]), r("row-2", ["zombie"])],
      hidden: ["dead"],
    };
    const next = reconcileLayout(stored, ["a", "b"]);
    expect(next.rows[0]).toEqual(r("row-1", ["a"]));
    expect(next.rows[next.rows.length - 1].panels).toEqual(["b"]);
    expect(next.hidden).toEqual([]);
  });

  it("dedupes panels that appear in multiple rows", () => {
    const stored = {
      rows: [r("row-1", ["a", "b"]), r("row-2", ["b", "c"])],
      hidden: [],
    };
    const next = reconcileLayout(stored, ["a", "b", "c"]);
    expect(next.rows.flatMap((row) => row.panels)).toEqual(["a", "b", "c"]);
  });
});

describe("applyReorder", () => {
  it("merges into the target row before the over-panel on left drop", () => {
    const rows = [r("row-1", ["a"]), r("row-2", ["b", "c"])];
    const next = applyReorder(rows, "a", "c", "left");
    expect(next.length).toBe(1);
    expect(next[0].panels).toEqual(["b", "a", "c"]);
  });

  it("merges into the target row after the over-panel on right drop", () => {
    const rows = [r("row-1", ["a"]), r("row-2", ["b"])];
    const next = applyReorder(rows, "a", "b", "right");
    expect(next.length).toBe(1);
    expect(next[0].panels).toEqual(["b", "a"]);
  });

  it("creates a new row above on top drop", () => {
    const rows = [r("row-1", ["a", "b"]), r("row-2", ["c"])];
    const next = applyReorder(rows, "b", "c", "top");
    expect(next.map((row) => row.panels)).toEqual([["a"], ["b"], ["c"]]);
  });

  it("creates a new row below on bottom drop", () => {
    const rows = [r("row-1", ["a"]), r("row-2", ["b", "c"])];
    const next = applyReorder(rows, "c", "a", "bottom");
    expect(next.map((row) => row.panels)).toEqual([["a"], ["c"], ["b"]]);
  });

  it("removes empty source rows", () => {
    const rows = [r("row-1", ["a"]), r("row-2", ["b"])];
    const next = applyReorder(rows, "a", "b", "left");
    expect(next.length).toBe(1);
    expect(next[0].panels).toEqual(["a", "b"]);
  });

  it("is a no-op when active equals over", () => {
    const rows = [r("row-1", ["a", "b"])];
    expect(applyReorder(rows, "a", "a", "left")).toBe(rows);
  });

  it("returns the same rows when target is missing", () => {
    const rows = [r("row-1", ["a"])];
    expect(applyReorder(rows, "a", "missing", "left")).toBe(rows);
  });
});
