/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Outcome of probing an MCP server (subprocess spawn or HTTP handshake).
 */
export type McpServerTestResponse = {
    ok: boolean;
    name: string;
    scope: McpServerTestResponse.scope;
    transport: string;
    latencyMs?: number;
    toolCount?: number;
    error?: (string | null);
};
export namespace McpServerTestResponse {
    export enum scope {
        NATIVE = 'native',
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

