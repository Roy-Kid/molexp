/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { McpOAuth2AuthRequest } from './McpOAuth2AuthRequest';
/**
 * Remote HTTP MCP server spec.
 *
 * Two transports: ``http`` (streamable HTTP, Claude Code convention)
 * and ``sse`` (legacy long-poll). Use ``http`` for any new server.
 */
export type McpHttpSpecRequest = {
    type: McpHttpSpecRequest.type;
    url: string;
    /**
     * Values may contain ${SECRET:KEY} placeholders.
     */
    headers?: Record<string, string>;
    /**
     * Optional structured auth. When set, the runtime drives the OAuth flow and ignores any 'Authorization' header here.
     */
    auth?: (McpOAuth2AuthRequest | null);
};
export namespace McpHttpSpecRequest {
    export enum type {
        HTTP = 'http',
        SSE = 'sse',
    }
}

