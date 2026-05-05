/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MCPHttpSpecRequest } from './MCPHttpSpecRequest';
import type { MCPStdioSpecRequest } from './MCPStdioSpecRequest';
/**
 * Create or replace an MCP server entry at the chosen scope.
 */
export type MCPServerUpsertRequest = {
    /**
     * Server name; lowercase letters, digits, underscore, hyphen; must start with a letter or digit.
     */
    name: string;
    /**
     * VSCode-style scope. Workspace overrides User on name collision.
     */
    scope?: MCPServerUpsertRequest.scope;
    spec: (MCPStdioSpecRequest | MCPHttpSpecRequest);
};
export namespace MCPServerUpsertRequest {
    /**
     * VSCode-style scope. Workspace overrides User on name collision.
     */
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

