/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Whether the named server currently has a usable OAuth token on disk.
 *
 * ``hasTokens`` is True after a successful Connect; False if the user has
 * never connected, has disconnected, or the token file got corrupted.
 */
export type McpOAuthStatusResponse = {
    name: string;
    scope: McpOAuthStatusResponse.scope;
    hasTokens: boolean;
    scopes?: Array<string>;
};
export namespace McpOAuthStatusResponse {
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

