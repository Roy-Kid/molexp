import { describe, expect, it } from "@rstest/core";
import { AGENT_SETTINGS_TABS } from "./tabs";

describe("Agent settings navigation", () => {
  it("uses capability-level Claude Code-style sections", () => {
    expect(AGENT_SETTINGS_TABS.map((tab) => tab.value)).toEqual([
      "agents",
      "model",
      "instructions",
      "skills",
      "mcp",
    ]);
  });

  it("opens on the static agents overview before network-backed settings", () => {
    expect(AGENT_SETTINGS_TABS[0]).toMatchObject({
      value: "agents",
      contentKey: "agents-overview",
    });
  });

  it("keeps tools inside their owning MCP server instead of a detached tab", () => {
    expect(AGENT_SETTINGS_TABS[AGENT_SETTINGS_TABS.length - 1]).toMatchObject({
      value: "mcp",
      contentKey: "mcp-servers",
    });
  });
});
