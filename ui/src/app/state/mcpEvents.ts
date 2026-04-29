/**
 * MCP-config-changed bus.
 *
 * The Tools panel and the MCP Servers tab live in different React subtrees;
 * lifting state up would force the whole AgentSettingsViewer to know about
 * tool fetching. A tiny window-event bus keeps the coupling at zero —
 * whoever cares about MCP config changes (today: ToolsTab) listens; whoever
 * mutates it (today: McpServersTab) emits.
 *
 * Same-window only: ``CustomEvent`` doesn't cross tabs. That's fine — every
 * change that matters is initiated from this window.
 */

export const MCP_CONFIG_CHANGED_EVENT = "molexp:mcp-config-changed";

export const emitMcpConfigChanged = (): void => {
  window.dispatchEvent(new CustomEvent(MCP_CONFIG_CHANGED_EVENT));
};

export const onMcpConfigChanged = (handler: () => void): (() => void) => {
  window.addEventListener(MCP_CONFIG_CHANGED_EVENT, handler);
  return () => window.removeEventListener(MCP_CONFIG_CHANGED_EVENT, handler);
};
