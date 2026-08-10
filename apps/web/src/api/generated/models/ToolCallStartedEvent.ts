/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Emitted when the emergent loop dispatches a tool call.
 *
 * ``args_summary`` is a short human-readable rendering of the call
 * arguments — never the full payload, so the event stream stays cheap.
 */
export type ToolCallStartedEvent = {
    args_summary?: string;
    kind?: string;
    timestamp?: string;
    tool_name: string;
};

