/**
 * Top-level tabs for AgentSettingsViewer.
 *
 * Capability surfaces: model (with a short Chat/Plan overview on top),
 * persistent instructions, reusable skills, and MCP servers. Chat vs Plan
 * is a **composer mode** (Shift+Tab), not a separate settings section —
 * tools live under their owning MCP server row.
 *
 * Pure data so it can be unit-tested without JSX / lucide / api clients.
 * The renderer maps each `contentKey` to a React component.
 */

export type AgentSettingsTabKey = "model" | "instructions" | "skills" | "mcp";

export interface AgentSettingsTabDef {
  /** URL-safe slug used by EntityPage for tab routing. */
  readonly value: AgentSettingsTabKey;
  /** Human-visible label rendered in the tab strip. */
  readonly label: string;
  /** Which content component the renderer mounts for this tab. */
  readonly contentKey: "providers-form" | "instructions-form" | "skills-list" | "mcp-servers";
}

export const AGENT_SETTINGS_TABS: readonly AgentSettingsTabDef[] = [
  { value: "model", label: "Model", contentKey: "providers-form" },
  { value: "instructions", label: "Instructions", contentKey: "instructions-form" },
  { value: "skills", label: "Skills", contentKey: "skills-list" },
  { value: "mcp", label: "MCP", contentKey: "mcp-servers" },
];
