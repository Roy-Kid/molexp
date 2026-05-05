/**
 * Top-level tabs for AgentSettingsViewer.
 *
 * Per the agent-harness UI lockstep spec (§8) the legacy 5-tab layout
 * collapses to three semantic groups:
 *
 *   - "agent"        — agent-core configuration: instructions, slash
 *                       commands, native tools.
 *   - "providers"    — model provider settings (registry-driven).
 *   - "tool-sources" — pluggable tool sources (today: MCP servers).
 *
 * The descriptor array is pure data so it can be unit-tested from
 * the node test environment without pulling in JSX, lucide, or the
 * api-client singleton chain.
 *
 * The renderer (`AgentSettingsViewer`) maps each `contentKey` to a
 * concrete React component and constructs the full tab descriptors
 * for `EntityPage`.
 */

export type AgentSettingsTabKey = "agent" | "providers" | "tool-sources";

export interface AgentSettingsTabDef {
  /** URL-safe slug used by EntityPage for tab routing. */
  readonly value: AgentSettingsTabKey;
  /** Human-visible label rendered in the tab strip. */
  readonly label: string;
  /** Which content component the renderer mounts for this tab. */
  readonly contentKey: "agent-core" | "providers-form" | "mcp-servers";
}

export const AGENT_SETTINGS_TABS: readonly AgentSettingsTabDef[] = [
  { value: "agent", label: "Agent", contentKey: "agent-core" },
  { value: "providers", label: "Model providers", contentKey: "providers-form" },
  { value: "tool-sources", label: "Tool sources", contentKey: "mcp-servers" },
];
