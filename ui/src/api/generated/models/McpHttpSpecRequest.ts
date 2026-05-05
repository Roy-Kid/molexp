/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MCPOAuth2AuthRequest } from './MCPOAuth2AuthRequest';
/**
 * Remote HTTP MCP server spec.
 *
 * Two transports: ``http`` (streamable HTTP, Claude Code convention)
 * and ``sse`` (legacy long-poll). Use ``http`` for any new server.
 */
export type MCPHttpSpecRequest = {
    type: MCPHttpSpecRequest.type;
    url: string;
    /**
     * Values may contain ${SECRET:KEY} placeholders.
     */
    headers?: Record<string, string>;
    /**
     * Optional structured auth. When set, the runtime drives the OAuth flow and ignores any 'Authorization' header here.
     */
    auth?: (MCPOAuth2AuthRequest | null);
};
export namespace MCPHttpSpecRequest {
    export enum type {
        HTTP = 'http',
        SSE = 'sse',
    }
}

