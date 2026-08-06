/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One MCP server row (no secret values — names/refs only).
 */
export type McpServerResponse = {
    args: Array<string>;
    auth?: (Record<string, any> | null);
    command: (string | null);
    env?: Record<string, string>;
    envKeys: Array<string>;
    headerKeys: Array<string>;
    invalidReason: string;
    knowledgeSources?: Array<string>;
    name: string;
    scope: string;
    secretRefs: Array<string>;
    shadowed: boolean;
    transport: string;
    unresolvedSecrets: Array<string>;
    url: (string | null);
    valid: boolean;
};

