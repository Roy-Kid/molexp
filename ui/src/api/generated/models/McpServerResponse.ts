/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MCPAuthSummary } from './MCPAuthSummary';
/**
 * One MCP server entry, possibly merged across scopes.
 *
 * ``shadowed`` is True when this entry exists at User scope but is
 * overridden by a Workspace entry of the same name. ``unresolvedSecrets``
 * lists ``${SECRET:KEY}`` references that have no value in either secret
 * store — the runtime skips such entries.
 */
export type MCPServerResponse = {
    name: string;
    scope: MCPServerResponse.scope;
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
    auth?: (MCPAuthSummary | null);
};
export namespace MCPServerResponse {
    export enum scope {
        USER = 'user',
        WORKSPACE = 'workspace',
    }
}

