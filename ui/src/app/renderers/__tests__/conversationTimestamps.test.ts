/**
 * Tests for the internal-steps timestamp de-duplication: the pure
 * visibleTimestampFlags helper is exercised directly, and the JSX wiring in
 * agent/conversation.tsx is asserted as a source contract (rstest node env,
 * no jsdom — see app/runs/metrics/RunMetricsView.test.tsx).
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

import { visibleTimestampFlags } from "../agentEvents";

const ts = (value: string): { ts: string } => ({ ts: value });

describe("visibleTimestampFlags", () => {
  it("shows only the first of an adjacent run of identical timestamps", () => {
    const events = [
      ts("2026-07-01T10:00:00Z"),
      ts("2026-07-01T10:00:00Z"),
      ts("2026-07-01T10:00:00Z"),
    ];
    expect(visibleTimestampFlags(events)).toEqual([true, false, false]);
  });

  it("shows every timestamp when they all differ", () => {
    const events = [ts("t1"), ts("t2"), ts("t3")];
    expect(visibleTimestampFlags(events)).toEqual([true, true, true]);
  });

  it("re-shows a timestamp after the value changes (adjacent-only dedupe)", () => {
    const events = [ts("t1"), ts("t1"), ts("t2"), ts("t1")];
    expect(visibleTimestampFlags(events)).toEqual([true, false, true, true]);
  });

  it("handles the empty list", () => {
    expect(visibleTimestampFlags([])).toEqual([]);
  });
});

describe("conversation.tsx wiring (source contract)", () => {
  const src = readFileSync(
    resolve(dirname(fileURLToPath(import.meta.url)), "../agent/conversation.tsx"),
    "utf8",
  );

  it("InternalSteps derives per-row visibility from visibleTimestampFlags", () => {
    expect(src).toContain("visibleTimestampFlags(detailSteps)");
    expect(src).toContain("showTimestamp={row.showTimestamp}");
  });

  it("suppressed rows keep the full timestamp on a hover tooltip", () => {
    expect(src).toContain("title={event.ts}");
    expect(src).toContain("showTimestamp ? formatTs(event.ts) : null");
  });
});
