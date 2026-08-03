import { describe, expect, it } from "@rstest/core";

import { normalizeRunStatus } from "./RunStatusBadge";

describe("normalizeRunStatus", () => {
  it("keeps the MolCrafts status vocabulary unchanged", () => {
    expect(normalizeRunStatus("queued")).toBe("queued");
    expect(normalizeRunStatus("running")).toBe("running");
    expect(normalizeRunStatus("completed")).toBe("completed");
    expect(normalizeRunStatus("failed")).toBe("failed");
  });

  it("normalizes wire aliases at the presentation boundary", () => {
    expect(normalizeRunStatus("pending")).toBe("queued");
    expect(normalizeRunStatus("succeeded")).toBe("completed");
    expect(normalizeRunStatus("error")).toBe("failed");
    expect(normalizeRunStatus("skipped")).toBe("cancelled");
    expect(normalizeRunStatus("active")).toBe("ready");
    expect(normalizeRunStatus("waiting_for_review")).toBe("warning");
    expect(normalizeRunStatus("awaiting_approval")).toBe("warning");
    expect(normalizeRunStatus("awaiting_user")).toBe("warning");
    expect(normalizeRunStatus("paused")).toBe("warning");
  });

  it("uses a neutral canonical fallback for an unknown wire value", () => {
    expect(normalizeRunStatus("future-status")).toBe("ready");
    expect(normalizeRunStatus(null)).toBeNull();
  });
});
