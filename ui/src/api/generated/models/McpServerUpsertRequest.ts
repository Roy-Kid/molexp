/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { McpHttpSpecRequest } from './McpHttpSpecRequest';
import type { McpStdioSpecRequest } from './McpStdioSpecRequest';
/**
 * Create or replace an MCP server entry at the chosen scope.
 */
export type McpServerUpsertRequest = {
    /**
     * Server name; lowercase letters, digits, underscore, hyphen; must start with a letter or digit.
     */
    name: string;
    /**
     * VSCode-style scope. Workspace overrides User on name collision.
     */
    scope?: McpServerUpsertRequest.scope;
    spec: (McpStdioSpecRequest | McpHttpSpecRequest);
};
export namespace McpServerUpsertRequest {
    /**
     * VSCode-style scope. Workspace overrides User on name collision.
     */
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}
