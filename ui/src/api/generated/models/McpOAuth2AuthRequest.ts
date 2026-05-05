/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * OAuth 2.0 (Authorization Code + PKCE) auth for an HTTP MCP server.
 *
 * The actual token exchange happens via the dedicated /oauth* endpoints;
 * this is just the *intent* persisted in the spec. Empty ``scopes`` means
 * "let the IdP pick". ``clientId`` is optional and only set when the
 * target IdP doesn't support Dynamic Client Registration.
 */
export type McpOAuth2AuthRequest = {
    type: string;
    scopes?: Array<string>;
    /**
     * Pre-registered client_id; leave null to use Dynamic Client Registration.
     */
    clientId?: (string | null);
};

