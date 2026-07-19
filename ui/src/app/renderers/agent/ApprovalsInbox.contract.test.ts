import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "@rstest/core";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "ApprovalsInbox.tsx"), "utf8");

describe("ApprovalsInbox three-action contract", () => {
  it("offers Approve, Reject, and Revise", () => {
    expect(source).toContain("Approve");
    expect(source).toContain("Reject");
    expect(source).toContain("Revise");
    expect(source).toContain('action: "approve" | "reject" | "revise"');
  });

  it("sends fieldValues on revise", () => {
    expect(source).toContain("fieldValues");
    expect(source).toContain('action === "revise"');
  });

  it("hosts ReviewSurface for form packs", () => {
    expect(source).toContain("ReviewSurface");
    expect(source).toContain("formDocument");
  });

  it("calls ApprovalsService decide endpoint", () => {
    expect(source).toContain("decideApprovalApiApprovalsTaskKindTaskIdDecisionsPost");
  });
});
