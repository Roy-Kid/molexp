import { describe, expect, it } from "@rstest/core";
import { statusIconMeta } from "@/app/components/entity";

describe("statusIconMeta", () => {
  it("maps terminal success statuses to the success tone", () => {
    expect(statusIconMeta("succeeded").tone).toBe("success");
    expect(statusIconMeta("completed").tone).toBe("success");
  });

  it("maps failed statuses to the error tone", () => {
    expect(statusIconMeta("failed").tone).toBe("error");
    expect(statusIconMeta("timed_out").tone).toBe("error");
  });

  it("marks running as animated info", () => {
    const meta = statusIconMeta("running");
    expect(meta.tone).toBe("running");
    expect(meta.spin).toBe(true);
  });

  it("keeps pending and skipped visually quiet", () => {
    expect(statusIconMeta("pending").tone).toBe("neutral");
    expect(statusIconMeta("skipped").tone).toBe("neutral");
  });
});
