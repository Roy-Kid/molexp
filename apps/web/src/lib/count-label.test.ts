/**
 * Tests for the countLabel pluralization helper.
 */

import { describe, expect, it } from "@rstest/core";

import { countLabel } from "@/lib/count-label";

describe("countLabel", () => {
  it("uses the singular for exactly one", () => {
    expect(countLabel(1, "run")).toBe("1 run");
    expect(countLabel(1, "exp")).toBe("1 exp");
  });

  it("pluralizes zero and many", () => {
    expect(countLabel(0, "run")).toBe("0 runs");
    expect(countLabel(3, "exp")).toBe("3 exps");
  });

  it("accepts an explicit plural", () => {
    expect(countLabel(2, "study", "studies")).toBe("2 studies");
  });
});
