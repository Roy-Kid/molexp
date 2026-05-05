/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Set or clear an MCP secret value at the chosen scope.
 *
 * The plaintext ``value`` is sent up only; the secret store never returns
 * it via any GET endpoint. Pass an empty string to delete the key.
 */
export type McpSecretSetRequest = {
    /**
     * Plaintext value; empty deletes the key.
     */
    value?: string;
    /**
     * Where to write the secret. Workspace beats User on lookup.
     */
    scope?: McpSecretSetRequest.scope;
};
export namespace McpSecretSetRequest {
    /**
     * Where to write the secret. Workspace beats User on lookup.
     */
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

