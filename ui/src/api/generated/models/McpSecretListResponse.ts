/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { McpSecretRefRow } from './McpSecretRefRow';
/**
 * Secrets at the requested scope. Plaintext values are never returned.
 */
export type McpSecretListResponse = {
    scope: McpSecretListResponse.scope;
    path: string;
    secrets?: Array<McpSecretRefRow>;
};
export namespace McpSecretListResponse {
    export enum scope {
        NATIVE = 'native',
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}
