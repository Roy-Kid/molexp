import { describe, expect, it } from "@rstest/core";
import { LAND_NO_MESSAGE, LAND_YES_MESSAGE, looksLikeLandOffer } from "./LandDecisionBar";

describe("looksLikeLandOffer", () => {
  it("detects English archive/land offers", () => {
    expect(looksLikeLandOffer("Work finished. Archive this onto a formal experiment / run?")).toBe(
      true,
    );
    expect(looksLikeLandOffer("Should I land this into a run?")).toBe(true);
  });

  it("ignores unrelated answers", () => {
    expect(looksLikeLandOffer("Here is the Rg plot.")).toBe(false);
  });
});

describe("land decision messages", () => {
  it("posts English user replies", () => {
    expect(LAND_YES_MESSAGE).toMatch(/Yes/i);
    expect(LAND_YES_MESSAGE).toMatch(/archive/i);
    expect(LAND_NO_MESSAGE).toMatch(/No/i);
    expect(LAND_NO_MESSAGE).toMatch(/scratch/i);
  });
});
