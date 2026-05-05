import { describe, expect, it } from "@rstest/core";

import { AGENT_SETTINGS_TABS } from "../agent_settings/tabs";

describe("AGENT_SETTINGS_TABS", () => {
  // ac-003 — settings view splits into Agent / Model providers / Tool sources tabs.
  it("declares exactly three tabs", () => {
    expect(AGENT_SETTINGS_TABS).toHaveLength(3);
  });

  it("labels the three tabs Agent / Model providers / Tool sources", () => {
    expect(AGENT_SETTINGS_TABS.map((t) => t.label)).toEqual([
      "Agent",
      "Model providers",
      "Tool sources",
    ]);
  });

  it("uses stable URL-safe values for each tab", () => {
    expect(AGENT_SETTINGS_TABS.map((t) => t.value)).toEqual(["agent", "providers", "tool-sources"]);
  });

  it("binds the MCP servers content only to the Tool sources tab", () => {
    // ac-003 — the MCP servers panel must not appear under Agent or
    // Model providers. Using the contentKey marker keeps this test
    // node-environment-friendly (no JSX walk, no React import).
    const withMcp = AGENT_SETTINGS_TABS.filter((t) => t.contentKey === "mcp-servers");
    expect(withMcp).toHaveLength(1);
    expect(withMcp[0].value).toBe("tool-sources");
  });

  it("does not bind any tab to a legacy single-purpose key", () => {
    // The legacy 5-tab layout (provider / instructions / commands /
    // tools / mcp) is gone — every contentKey must be one of the
    // three new semantic groups.
    const legacy = ["provider", "instructions", "commands", "tools", "mcp"];
    for (const t of AGENT_SETTINGS_TABS) {
      expect(legacy.includes(t.value as string)).toBe(false);
    }
  });
});
