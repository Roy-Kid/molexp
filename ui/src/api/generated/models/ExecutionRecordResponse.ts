/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One execution attempt of a Run.
 *
 * Mirrors :class:`molexp.workspace.models.ExecutionRecord` with
 * JSON-friendly field names so the UI can render a per-attempt
 * timeline.
 */
export type ExecutionRecordResponse = {
    executionId: string;
    finishedAt?: (string | null);
    schedulerJobId?: (string | null);
    startedAt: string;
    status: string;
};

