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
export type MCPOAuthStartResponse = {
    name: string;
    scope: MCPOAuthStartResponse.scope;
    authorizeUrl: string;
};
export namespace MCPOAuthStartResponse {
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

