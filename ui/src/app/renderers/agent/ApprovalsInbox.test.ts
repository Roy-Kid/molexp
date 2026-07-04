/**
 * Approvals-inbox wiring contract (vision-loop-01).
 *
 * Rstest runs in a node environment without jsdom, so — per repo convention
 * (see DeliverablesPanel.test.ts) — the JSX wiring is asserted against the
 * component source: the inbox must consume the generated ApprovalsService
 * (list + decide), refetch off the SSE ping stream (and close it on unmount),
 * `waiting_approval` must render as a warning (never an error), and both the
 * Agents-hub landing and the plan composer must mount the inbox.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const read = (relative: string): string => readFileSync(resolve(__dirname, relative), "utf-8");

const inbox = read("./ApprovalsInbox.tsx");
const statusBadge = read("../../components/entity/StatusBadge.tsx");
const agentViewer = read("../AgentViewer.tsx");
const planComposer = read("../../components/PlanComposer.tsx");

describe("ApprovalsInbox", () => {
  it("consumes the generated ApprovalsService for list + decide", () => {
    expect(inbox).toContain("ApprovalsService.listPendingApprovalsApiApprovalsGet()");
    expect(inbox).toContain(
      "ApprovalsService.decideApprovalApiApprovalsTaskKindTaskIdDecisionsPost",
    );
  });

  it("refetches off the SSE ping stream and closes it on unmount", () => {
    expect(inbox).toContain('new EventSource("/api/approvals/events")');
    expect(inbox).toContain('addEventListener("changed"');
    expect(inbox).toContain("source.close()");
  });

  it("offers Approve and Reject (reject with an optional reason)", () => {
    expect(inbox).toContain("Approve");
    expect(inbox).toContain("Reject");
    expect(inbox).toContain("granted, reason: reason.trim() || undefined");
  });

  it("renders the requested-at moment through the shared formatDateTime", () => {
    expect(inbox).toContain("formatDateTime(item.requestedAt)");
  });
});

describe("waiting_approval status surface", () => {
  it("maps waiting_approval to a warning tone, never an error", () => {
    expect(statusBadge).toContain('waiting_approval: "warning"');
  });

  it("mounts the inbox on the Agents-hub landing", () => {
    expect(agentViewer).toContain("<ApprovalsInbox onDecided={onRefresh} />");
  });

  it("treats a suspended plan task as in-flight and decides inline", () => {
    expect(planComposer).toContain('task?.status === "waiting_approval"');
    expect(planComposer).toContain("<ApprovalsInbox taskId={taskId} />");
    // Polling must continue through suspension so the resumed pipeline lands.
    expect(planComposer).toContain("isRunning || isWaitingApproval");
  });
});
