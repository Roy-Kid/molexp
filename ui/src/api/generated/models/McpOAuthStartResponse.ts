/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Result of POST /mcp/servers/{name}/oauth/start.
 *
 * The UI opens ``authorizeUrl`` in a popup; once the IdP bounces back to
 * the SPA the SPA POSTs ``code``+``state`` to the callback endpoint to
 * finish the flow.
 */
export type McpOAuthStartResponse = {
    name: string;
    scope: McpOAuthStartResponse.scope;
    authorizeUrl: string;
};
export namespace McpOAuthStartResponse {
    export enum scope {
        NATIVE = 'native',
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}
