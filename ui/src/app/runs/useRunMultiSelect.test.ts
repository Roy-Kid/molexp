/**
 * RED tests for the multi-run selection transition (ac-008).
 *
 * The shift-range / ctrl-meta-toggle logic is extracted into a pure reducer
 * `nextSelection` so it is testable under the node environment without rendering
 * the hook. Import fails today because useRunMultiSelect.ts does not exist (RED).
 */

import { describe, expect, it } from "@rstest/core";

import { type MultiSelectState, nextSelection } from "./useRunMultiSelect";

const IDS = ["r0", "r1", "r2", "r3", "r4"];
const empty = (): MultiSelectState => ({ selected: new Set<string>(), anchor: null });

const selectedArray = (state: MultiSelectState): string[] => Array.from(state.selected).sort();

describe("nextSelection (ac-008)", () => {
  it("plain click selects only the clicked row and sets the anchor", () => {
    const next = nextSelection(empty(), 2, IDS, { shift: false, meta: false });
    expect(selectedArray(next)).toEqual(["r2"]);
    expect(next.anchor).toBe(2);
  });

  it("ctrl/meta click toggles a single row without clearing others", () => {
    const a = nextSelection(empty(), 1, IDS, { shift: false, meta: false }); // {r1}
    const b = nextSelection(a, 3, IDS, { shift: false, meta: true }); // add r3
    expect(selectedArray(b)).toEqual(["r1", "r3"]);
    const c = nextSelection(b, 1, IDS, { shift: false, meta: true }); // toggle r1 off
    expect(selectedArray(c)).toEqual(["r3"]);
  });

  it("shift click selects the inclusive range from the anchor", () => {
    const anchored = nextSelection(empty(), 1, IDS, { shift: false, meta: false }); // anchor 1
    const ranged = nextSelection(anchored, 3, IDS, { shift: true, meta: false });
    expect(selectedArray(ranged)).toEqual(["r1", "r2", "r3"]);
    // anchor is preserved so a subsequent shift re-ranges from the same origin
    expect(ranged.anchor).toBe(1);
  });

  it("shift range works backwards (click before the anchor)", () => {
    const anchored = nextSelection(empty(), 3, IDS, { shift: false, meta: false }); // anchor 3
    const ranged = nextSelection(anchored, 1, IDS, { shift: true, meta: false });
    expect(selectedArray(ranged)).toEqual(["r1", "r2", "r3"]);
  });

  it("shift with no anchor falls back to single selection", () => {
    const next = nextSelection(empty(), 2, IDS, { shift: true, meta: false });
    expect(selectedArray(next)).toEqual(["r2"]);
    expect(next.anchor).toBe(2);
  });
});
