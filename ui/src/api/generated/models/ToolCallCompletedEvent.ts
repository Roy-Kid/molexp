/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Emitted when a dispatched tool call returns.
 *
 * ``ok`` is ``False`` when the tool raised / returned a retry prompt;
 * ``result_summary`` is a short rendering of the return value.
 */
export type ToolCallCompletedEvent = {
    kind?: string;
    ok?: boolean;
    result_summary?: string;
    timestamp?: string;
    tool_name: string;
};

