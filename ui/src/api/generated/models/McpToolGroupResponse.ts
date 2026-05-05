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
export type MCPToolGroupResponse = {
    server: string;
    scope: MCPToolGroupResponse.scope;
    ok: boolean;
    toolCount?: number;
    error?: (string | null);
};
export namespace MCPToolGroupResponse {
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

