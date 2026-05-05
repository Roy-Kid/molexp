/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Read-only view of a user/workspace HTTP-webhook tool's wiring.
 *
 * Header values are returned **with secret references intact**
 * (``${SECRET:KEY}``); the actual secret value never leaves the
 * server.
 */
export type CustomToolHttpInvokerResponse = {
    kind?: string;
    url: string;
    method?: CustomToolHttpInvokerResponse.method;
    headers?: Record<string, string>;
    bodyTemplate?: string;
};
export namespace CustomToolHttpInvokerResponse {
    export enum method {
        GET = 'GET',
        POST = 'POST',
        PUT = 'PUT',
        DELETE = 'DELETE',
    }
}

