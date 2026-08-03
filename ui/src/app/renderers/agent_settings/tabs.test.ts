import { describe, expect, it } from "@rstest/core";
import { AGENT_SETTINGS_TABS } from "./tabs";

describe("Agent settings navigation", () => {
  it("uses capability-level sections without a separate Agents tab", () => {
    expect(AGENT_SETTINGS_TABS.map((tab) => tab.value)).toEqual([
      "model",
      "instructions",
      "skills",
      "mcp",
    ]);
  });

  it("opens on Model (mode overview + providers live there)", () => {
    expect(AGENT_SETTINGS_TABS[0]).toMatchObject({
      value: "model",
      contentKey: "providers-form",
    });
  });

  it("keeps tools inside their owning MCP server instead of a detached tab", () => {
    expect(AGENT_SETTINGS_TABS[AGENT_SETTINGS_TABS.length - 1]).toMatchObject({
      value: "mcp",
      contentKey: "mcp-servers",
    });
  });
});
