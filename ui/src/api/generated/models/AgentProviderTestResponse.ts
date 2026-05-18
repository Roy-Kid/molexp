/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Result of probing the configured provider with a minimal request.
 *
 * ``ok=True`` means we got a model response back. ``latencyMs`` is the
 * wall-clock RTT for the probe; ``error`` is filled only on failure
 * with a short, user-readable description (no stack trace, no key).
 */
export type AgentProviderTestResponse = {
    ok?: boolean;
    provider?: string;
    model?: string;
    latencyMs?: number;
    reply?: string;
    error?: (string | null);
};
