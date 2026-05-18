/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { McpServerResponse } from './McpServerResponse';
/**
 * Merged view of both scopes plus the resolved file paths.
 *
 * ``workspacePath`` and ``userPath`` are the absolute paths the store
 * would read/write at each scope (whether or not the file currently
 * exists) — useful for UI tooltips like "Edit ~/.molexp/mcp.json".
 */
export type McpServerListResponse = {
    workspacePath: string;
    userPath: string;
    servers?: Array<McpServerResponse>;
};
