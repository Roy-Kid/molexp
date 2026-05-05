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
export type MCPSecretSetRequest = {
    /**
     * Plaintext value; empty deletes the key.
     */
    value?: string;
    /**
     * Where to write the secret. Workspace beats User on lookup.
     */
    scope?: MCPSecretSetRequest.scope;
};
export namespace MCPSecretSetRequest {
    /**
     * Where to write the secret. Workspace beats User on lookup.
     */
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

