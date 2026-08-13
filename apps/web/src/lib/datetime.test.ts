/**
 * `formatDateTime` is the single spelling for absolute timestamps in the UI
 * (see datetime.ts). Pure-function tests — timezone-safe by feeding local
 * (offset-free) ISO strings, which `new Date` interprets in the local zone,
 * exactly matching the formatter's local-time output.
 */

import { describe, expect, it } from "@rstest/core";

import { formatDateTime } from "./datetime";

describe("formatDateTime", () => {
  it("renders a local ISO timestamp as YYYY-MM-DD HH:mm", () => {
    expect(formatDateTime("2026-07-02T18:16:01")).toBe("2026-07-02 18:16");
  });

  it("drops sub-minute precision (the hover title carries the raw ISO)", () => {
    expect(formatDateTime("2026-07-02T18:16:59.039865")).toBe("2026-07-02 18:16");
  });

  it("zero-pads month, day, hour, and minute", () => {
    expect(formatDateTime("2026-01-05T08:04:00")).toBe("2026-01-05 08:04");
  });

  it("returns an em dash for missing values", () => {
    expect(formatDateTime(null)).toBe("—");
    expect(formatDateTime(undefined)).toBe("—");
    expect(formatDateTime("")).toBe("—");
  });

  it("echoes unparseable input instead of hiding it", () => {
    expect(formatDateTime("not-a-date")).toBe("not-a-date");
  });

  it("accepts zoned timestamps (rendered in the local zone)", () => {
    const iso = "2026-07-02T10:16:00Z";
    const d = new Date(iso);
    const pad = (n: number): string => String(n).padStart(2, "0");
    const expected = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(
      d.getHours(),
    )}:${pad(d.getMinutes())}`;
    expect(formatDateTime(iso)).toBe(expected);
  });
});
