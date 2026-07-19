/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ToolParameterResponse } from './ToolParameterResponse';
/**
 * One tool exposed by an MCP server.
 *
 * ``source`` is ``"mcp:<server-name>"`` so the UI can attach the tool to
 * its owning server's expanded row.
 */
export type AgentToolResponse = {
    description?: string;
    name: string;
    parameters?: Array<ToolParameterResponse>;
    requiresApproval?: boolean;
    source: string;
};

