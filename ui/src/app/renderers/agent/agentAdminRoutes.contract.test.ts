import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "@rstest/core";

/**
 * vision-loop-03 contract: the Settings page's hand-written fetch paths in
 * app/state/api.ts must be served by real backend routes — proven here
 * against the regenerated OpenAPI client (AgentAdminService), which only
 * contains operations the server actually declares. Source-contract style
 * (no jsdom in the rstest env — see ApprovalsInbox.test.ts).
 */

const read = (rel: string): string =>
  readFileSync(join(__dirname, "..", "..", "..", rel), "utf8");

describe("agent admin settings routes", () => {
  const apiTs = read("app/state/api.ts");
  const generated = read("api/generated/services/AgentAdminService.ts");

  it("api.ts calls the provider + mcp settings paths", () => {
    expect(apiTs).toContain('fetch("/api/agent/provider")');
    expect(apiTs).toContain('fetch("/api/agent/provider/test"');
    expect(apiTs).toContain('fetch("/api/agent/mcp/servers")');
  });

  it("the served OpenAPI surface declares the same paths (no more 503 stub)", () => {
    expect(generated).toContain("/api/agent/provider");
    expect(generated).toContain("/api/agent/provider/test");
    expect(generated).toContain("/api/agent/mcp/servers");
  });

  it("provider responses stay secret-safe: masked preview fields, never a key value", () => {
    const model = read("api/generated/models/ProviderResponse.ts");
    expect(model).toContain("apiKeyPreview");
    expect(model).toContain("apiKeySet");
    expect(model).not.toContain("apiKeyValue");
  });
});
