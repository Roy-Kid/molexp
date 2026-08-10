/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ToolParameterResponse } from './ToolParameterResponse';
/**
 * One agent tool — molexp **builtin** or MCP-discovered.
 *
 * ``source`` is:
 *
 * * ``"builtin"`` — always-on molexp tools (``workspace_ensure``,
 * ``run_land``, ``code_write``, …)
 * * ``"mcp:<server-name>"`` — tool from an MCP server, so the UI can
 * attach it to that server's expanded row
 */
export type AgentToolResponse = {
    description?: string;
    name: string;
    parameters?: Array<ToolParameterResponse>;
    requiresApproval?: boolean;
    source: string;
};

