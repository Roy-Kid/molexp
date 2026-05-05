/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { McpAuthSummary } from './McpAuthSummary';
/**
 * One MCP server entry, possibly merged across scopes.
 *
 * ``shadowed`` is True when this entry exists at User scope but is
 * overridden by a Workspace entry of the same name. ``unresolvedSecrets``
 * lists ``${SECRET:KEY}`` references that have no value in either secret
 * store — the runtime skips such entries.
 */
export type McpServerResponse = {
    name: string;
    scope: McpServerResponse.scope;
    transport?: string;
    command?: (string | null);
    args?: Array<string>;
    url?: (string | null);
    envKeys?: Array<string>;
    headerKeys?: Array<string>;
    secretRefs?: Array<string>;
    unresolvedSecrets?: Array<string>;
    shadowed?: boolean;
    valid?: boolean;
    invalidReason?: string;
    auth?: (McpAuthSummary | null);
};
export namespace McpServerResponse {
    export enum scope {
        NATIVE = 'native',
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

