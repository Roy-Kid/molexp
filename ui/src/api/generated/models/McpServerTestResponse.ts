/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Outcome of probing an MCP server (subprocess spawn or HTTP handshake).
 */
export type MCPServerTestResponse = {
    ok: boolean;
    name: string;
    scope: MCPServerTestResponse.scope;
    transport: string;
    latencyMs?: number;
    toolCount?: number;
    error?: (string | null);
};
export namespace MCPServerTestResponse {
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

