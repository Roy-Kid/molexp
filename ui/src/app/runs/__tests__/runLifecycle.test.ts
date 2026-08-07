import { describe, expect, it } from "@rstest/core";
import { canAnalyzeFailure, canHarvest, canStart } from "@/app/runs/runLifecycle";

describe("runLifecycle knowledge affordances", () => {
  it("exposes harvest on terminal statuses", () => {
    expect(canHarvest("succeeded")).toBe(true);
    expect(canHarvest("failed")).toBe(true);
    expect(canHarvest("running")).toBe(false);
  });

  it("exposes analyze-failure only on failed", () => {
    expect(canAnalyzeFailure("failed")).toBe(true);
    expect(canAnalyzeFailure("cancelled")).toBe(false);
    expect(canAnalyzeFailure("succeeded")).toBe(false);
  });

  it("keeps start pending-only", () => {
    expect(canStart("pending")).toBe(true);
    expect(canStart("failed")).toBe(false);
  });
});
