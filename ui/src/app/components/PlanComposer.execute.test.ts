/**
 * Plan-composer execute-tail contract (vision-loop-02).
 *
 * Rstest runs in a node environment without jsdom, so — per repo convention
 * (see ApprovalsInbox.test.ts) — the wiring is asserted against the component
 * source: the composer must expose the "Execute after approvals" opt-in with
 * HONEST copy (server host + the three named gates), thread
 * `execute`/`compute_target` into the createPlanTask request body, feed the
 * target select from the generated TargetsService, and badge execute tasks.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "@rstest/core";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const composer = readFileSync(resolve(__dirname, "./PlanComposer.tsx"), "utf-8");

describe("PlanComposer execute tail", () => {
  it("offers the execute opt-in with honest server-host + three-gates copy", () => {
    expect(composer).toContain("Execute after approvals");
    expect(composer).toContain("server host");
    expect(composer).toContain("experiment spec, plan review, execution review");
  });

  it("threads execute + compute_target into the createPlanTask body", () => {
    expect(composer).toContain("execute,");
    expect(composer).toContain("compute_target: computeTarget,");
  });

  it("feeds the target select from the generated TargetsService", () => {
    expect(composer).toContain("TargetsService.listTargetsEndpointApiTargetsGet()");
    // An empty registry means the local default — the select must not invent targets.
    expect(composer).toContain("No compute targets registered");
  });

  it("badges an execute task so the operator can tell it apart", () => {
    expect(composer).toContain("task?.execute && <Badge");
  });

  it("generated request/response models carry the new fields", () => {
    const request = readFileSync(
      resolve(__dirname, "../../api/generated/models/PlanTaskCreateRequest.ts"),
      "utf-8",
    );
    const response = readFileSync(
      resolve(__dirname, "../../api/generated/models/PlanTaskResponse.ts"),
      "utf-8",
    );
    expect(request).toContain("execute?: boolean");
    expect(request).toContain("compute_target?: (string | null)");
    expect(response).toContain("execute?: boolean");
  });
});
