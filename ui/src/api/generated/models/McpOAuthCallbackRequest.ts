/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * OAuth callback payload posted by the SPA after the IdP bounces back.
 *
 * The SPA owns the redirect-URI route (``/oauth-callback``); it pulls
 * ``code`` and ``state`` from the query string and forwards them here.
 */
export type MCPOAuthCallbackRequest = {
    code: string;
    state?: (string | null);
};

