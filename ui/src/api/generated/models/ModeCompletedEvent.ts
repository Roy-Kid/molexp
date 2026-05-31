/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Terminal event — carries the run's final text + optional result.
 *
 * ``result`` is the JSON-mode dump of the terminal
 * :class:`~molexp.agent.loop.AgentRunResult` (minus ``events`` to
 * avoid recursion); the harness reconstructs the typed result from
 * the accumulated stream.
 */
export type ModeCompletedEvent = {
    kind?: string;
    result?: (Record<string, any> | null);
    text: string;
    timestamp?: string;
};

