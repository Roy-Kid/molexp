/**
 * Tests for the internal-steps timestamp de-duplication (pure
 * visibleTimestampFlags helper).
 */

import { describe, expect, it } from "@rstest/core";

import { visibleTimestampFlags } from "../agentEvents";

const ts = (value: string): { ts: string } => ({ ts: value });

describe("visibleTimestampFlags", () => {
  it("re-shows a timestamp after the value changes (adjacent-only dedupe)", () => {
    const events = [ts("t1"), ts("t1"), ts("t2"), ts("t1")];
    expect(visibleTimestampFlags(events)).toEqual([true, false, true, true]);
  });

  it("handles the empty list", () => {
    expect(visibleTimestampFlags([])).toEqual([]);
  });
});
