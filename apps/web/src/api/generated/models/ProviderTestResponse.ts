/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * POST /provider/test result — an honest preflight, not a chat reply.
 */
export type ProviderTestResponse = {
    error?: (string | null);
    latencyMs: number;
    model: string;
    ok: boolean;
    provider: string;
    reply: string;
};

