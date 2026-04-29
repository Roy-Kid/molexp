/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Local subprocess MCP server spec.
 */
export type McpStdioSpecRequest = {
    type?: string;
    command: string;
    args?: Array<string>;
    /**
     * Values may contain ${SECRET:KEY} placeholders.
     */
    env?: Record<string, string>;
};

