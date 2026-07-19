/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Per-server discovery status for the MCP server list.
 *
 * Even when a server is offline / misconfigured / unauthorized we want
 * the UI to render *something* under that server's heading — a row with
 * the error keeps users oriented instead of silently dropping the group.
 */
export type McpToolGroupResponse = {
    error?: (string | null);
    ok: boolean;
    scope: McpToolGroupResponse.scope;
    server: string;
    toolCount?: number;
};
export namespace McpToolGroupResponse {
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

