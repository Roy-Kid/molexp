/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Whether the agent runtime is ready to start a new session.
 *
 * ``ready=False`` indicates a configuration problem the user can
 * resolve in Agent Settings (most commonly: no API key). ``source``
 * is one of ``"stored"`` (workspace config), ``"env"`` (process env
 * var), or ``"none"`` (not configured).
 */
export type AgentHealthResponse = {
    envVar?: string;
    model?: string;
    provider?: string;
    ready?: boolean;
    reason?: string;
    source?: string;
};

