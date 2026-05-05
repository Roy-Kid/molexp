/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Per-server status surface for the Tools panel.
 *
 * Even when a server is offline / misconfigured / unauthorized we want
 * the UI to render *something* under that server's heading — a row with
 * the error keeps users oriented instead of silently dropping the group.
 */
export type McpToolGroupResponse = {
    server: string;
    scope: McpToolGroupResponse.scope;
    ok: boolean;
    toolCount?: number;
    error?: (string | null);
};
export namespace McpToolGroupResponse {
    export enum scope {
        NATIVE = 'native',
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

