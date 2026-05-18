/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Public-safe view of a server's structured auth settings.
 *
 * Token values, refresh tokens, and client secrets are never exposed —
 * only metadata the UI needs to render the connection card. ``connected``
 * indicates the token store on disk has at least one persisted token
 * (rough proxy for "user has completed Connect at least once").
 */
export type McpAuthSummary = {
    type: string;
    scopes?: Array<string>;
    clientId?: (string | null);
    connected?: boolean;
};
