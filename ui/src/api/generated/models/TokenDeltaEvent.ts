/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Emitted for one token-level text increment from the emergent loop.
 *
 * :class:`~molexp.agent.loops.interactive.InteractiveLoop` yields one
 * of these per assistant text delta so a CLI / SSE consumer can render
 * the reply as it streams. v1 keeps these in the accumulated
 * :attr:`~molexp.agent.loop.AgentRunResult.events` stream unfiltered.
 */
export type TokenDeltaEvent = {
    kind?: string;
    text: string;
    timestamp?: string;
};

