/**
 * Top-level tabs for AgentSettingsViewer.
 *
 * Claude Code-style capability surfaces: agents, model, persistent
 * instructions, reusable skills, and MCP servers. Tools are exposed by
 * their owning MCP server and are shown inside that server's expanded row.
 * The descriptor array is pure data so it can be unit-tested from
 * the node test environment without pulling in JSX, lucide, or the
 * api-client singleton chain.
 *
 * The renderer (`AgentSettingsViewer`) maps each `contentKey` to a
 * concrete React component and constructs the full tab descriptors
 * for `EntityPage`.
 */

export type AgentSettingsTabKey = "agents" | "model" | "instructions" | "skills" | "mcp";

export interface AgentSettingsTabDef {
  /** URL-safe slug used by EntityPage for tab routing. */
  readonly value: AgentSettingsTabKey;
  /** Human-visible label rendered in the tab strip. */
  readonly label: string;
  /** Which content component the renderer mounts for this tab. */
  readonly contentKey:
    | "agents-overview"
    | "providers-form"
    | "instructions-form"
    | "skills-list"
    | "mcp-servers";
}

export const AGENT_SETTINGS_TABS: readonly AgentSettingsTabDef[] = [
  { value: "agents", label: "Agents", contentKey: "agents-overview" },
  { value: "model", label: "Model", contentKey: "providers-form" },
  { value: "instructions", label: "Instructions", contentKey: "instructions-form" },
  { value: "skills", label: "Skills", contentKey: "skills-list" },
  { value: "mcp", label: "MCP", contentKey: "mcp-servers" },
];
