import { describe, expect, it } from "@rstest/core";
import { statusIconMeta } from "@/app/components/entity";

describe("statusIconMeta", () => {
  it("maps terminal success statuses to the canonical completed tone", () => {
    expect(statusIconMeta("succeeded").tone).toBe("completed");
    expect(statusIconMeta("completed").tone).toBe("completed");
  });

  it("maps failed statuses to the canonical failed tone", () => {
    expect(statusIconMeta("failed").tone).toBe("failed");
    expect(statusIconMeta("timed_out").tone).toBe("failed");
  });

  it("marks running as animated info", () => {
    const meta = statusIconMeta("running");
    expect(meta.tone).toBe("running");
    expect(meta.spin).toBe(true);
  });

  it("keeps pending and skipped in their canonical quiet tones", () => {
    expect(statusIconMeta("pending").tone).toBe("queued");
    expect(statusIconMeta("skipped").tone).toBe("cancelled");
  });

  it("maps suspended agent states to warning", () => {
    expect(statusIconMeta("waiting_approval").tone).toBe("warning");
    expect(statusIconMeta("awaiting_user").tone).toBe("warning");
  });
});
