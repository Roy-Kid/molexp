import { describe, expect, it } from "@rstest/core";

import type { ApiAgentSession } from "@/app/types";
import {
  isLegacySession,
  LEGACY_SESSION_STATUS,
  legacyBadgeMeta,
} from "../agent_session/inspectorHelpers";

const baseSession = (overrides: Partial<ApiAgentSession> = {}): ApiAgentSession =>
  ({
    sessionId: "s-1",
    taskId: "t-1",
    goal: "Test session",
    status: "running",
    createdAt: "2026-05-05T00:00:00Z",
    events: [],
    ...overrides,
  }) as unknown as ApiAgentSession;

describe("isLegacySession", () => {
  // ac-006 — inspector reads .molexp-agent/ paths and badges legacy.
  it("returns false for null / undefined sessions", () => {
    expect(isLegacySession(null)).toBe(false);
    expect(isLegacySession(undefined)).toBe(false);
  });

  it("returns true only when status is exactly 'legacy'", () => {
    expect(isLegacySession(baseSession({ status: "running" }))).toBe(false);
    expect(isLegacySession(baseSession({ status: "completed" }))).toBe(false);
    expect(isLegacySession(baseSession({ status: LEGACY_SESSION_STATUS }))).toBe(true);
  });
});

describe("legacyBadgeMeta", () => {
  it("labels the badge read-only and explains the migration in the tooltip", () => {
    const meta = legacyBadgeMeta();
    expect(meta.label).toContain("read-only");
    expect(meta.label).toContain("legacy");
    expect(meta.tooltip).toContain(".molexp-agent");
  });
});
