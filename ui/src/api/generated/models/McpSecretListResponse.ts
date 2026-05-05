/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MCPSecretRefRow } from './MCPSecretRefRow';
/**
 * Secrets at the requested scope. Plaintext values are never returned.
 */
export type MCPSecretListResponse = {
    scope: MCPSecretListResponse.scope;
    path: string;
    secrets?: Array<MCPSecretRefRow>;
};
export namespace MCPSecretListResponse {
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

