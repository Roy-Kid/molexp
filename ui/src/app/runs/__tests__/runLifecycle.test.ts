import { describe, expect, it } from "@rstest/core";
import {
  canCancel,
  canHarvest,
  canResume,
  canRerun,
  canStart,
  isTerminalStatus,
  runPhase,
} from "@/app/runs/runLifecycle";

describe("runLifecycle", () => {
  it("maps each status to a disjoint phase", () => {
    expect(runPhase("pending")).toBe("pending");
    expect(runPhase("running")).toBe("running");
    expect(runPhase("failed")).toBe("retryable");
    expect(runPhase("cancelled")).toBe("retryable");
    expect(runPhase("succeeded")).toBe("succeeded");
    expect(runPhase("skipped")).toBe("other");
  });

  it("gates start / cancel / resume / rerun / harvest without overlap on primary verbs", () => {
    // pending: only start
    expect(canStart("pending")).toBe(true);
    expect(canCancel("pending")).toBe(false);
    expect(canResume("pending")).toBe(false);
    expect(canHarvest("pending")).toBe(false);

    // running: only cancel
    expect(canStart("running")).toBe(false);
    expect(canCancel("running")).toBe(true);
    expect(canResume("running")).toBe(false);
    expect(canHarvest("running")).toBe(false);

    // failed: resume+rerun+harvest, no start/cancel
    expect(canStart("failed")).toBe(false);
    expect(canCancel("failed")).toBe(false);
    expect(canResume("failed")).toBe(true);
    expect(canRerun("failed")).toBe(true);
    expect(canHarvest("failed")).toBe(true);

    // succeeded: harvest only
    expect(canStart("succeeded")).toBe(false);
    expect(canCancel("succeeded")).toBe(false);
    expect(canResume("succeeded")).toBe(false);
    expect(canHarvest("succeeded")).toBe(true);
  });

  it("treats skipped as terminal but not harvestable", () => {
    expect(isTerminalStatus("skipped")).toBe(true);
    expect(canHarvest("skipped")).toBe(false);
  });
});
