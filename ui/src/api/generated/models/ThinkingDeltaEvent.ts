/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Emitted for one reasoning-token increment from the emergent loop.
 *
 * The orchestration-level projection of a
 * :class:`~molexp.agent.router.ThinkingDeltaChunk`: a reasoning model's
 * private chain-of-thought, streamed *before* the answer.
 * :class:`~molexp.agent.loops.interactive.InteractiveLoop` yields one per
 * reasoning delta so a CLI / SSE consumer can surface "thinking…" in a
 * collapsed / dimmed treatment, kept distinct from the answer's
 * :class:`TokenDeltaEvent`\ s. A model that does not reason emits none.
 */
export type ThinkingDeltaEvent = {
    kind?: string;
    text: string;
    timestamp?: string;
};

