/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * HTTP-webhook invoker spec for a user-declared tool.
 */
export type CustomToolHttpInvokerRequest = {
    kind?: string;
    url: string;
    method?: CustomToolHttpInvokerRequest.method;
    /**
     * Outgoing headers; values may carry ${SECRET:KEY} placeholders resolved against the workspace secret store at request time.
     */
    headers?: Record<string, string>;
    /**
     * Optional template for the request body. Empty = send the LLM-supplied arguments as JSON.
     */
    bodyTemplate?: string;
};
export namespace CustomToolHttpInvokerRequest {
    export enum method {
        GET = 'GET',
        POST = 'POST',
        PUT = 'PUT',
        DELETE = 'DELETE',
    }
}

