/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Terminal event — the emergent outer loop parked itself durably.
 *
 * The dual of :class:`LoopCompletedEvent` for the suspend branch: a
 * :class:`~molexp.agent.loops.hooks.ShouldStopGuard` returned
 * :meth:`~molexp.agent.loops.hooks.HookOutcome.suspend`, so
 * :class:`~molexp.agent.loops.interactive.InteractiveLoop` stops without a
 * completion. No pending record is written — the session entry tree and its
 * ``leaf`` pointer (identified by :attr:`leaf_id`) are already durably
 * persisted, so a later turn resumes straight from that tip.
 *
 * Attributes:
 * reason: The guard's suspend token — a human-readable rationale.
 * leaf_id: The session's active tip at suspend time (a persisted entry
 * id), the durable resume anchor.
 */
export type LoopSuspendedEvent = {
    kind?: string;
    leaf_id?: string;
    reason?: string;
    timestamp?: string;
};

