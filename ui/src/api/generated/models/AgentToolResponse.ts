/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ToolParameterResponse } from './ToolParameterResponse';
/**
 * One tool exposed to the agent — native or MCP-discovered.
 *
 * For MCP tools, ``source`` is ``"mcp:<server-name>"`` so the UI can
 * group by server. Native tools keep ``source = "native"``.
 */
export type AgentToolResponse = {
    name: string;
    description?: string;
    parameters?: Array<ToolParameterResponse>;
    requiresApproval?: boolean;
    source?: string;
};

