/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One execution attempt of a run, surfaced for the workspace runs table.
 */
export type WorkspaceExecutionRow = {
    executionId: string;
    runId: string;
    status: string;
    startedAt: string;
    finishedAt?: (string | null);
    durationSeconds?: (number | null);
    schedulerJobId?: (string | null);
    backend?: (string | null);
    backendMetadata?: Record<string, string>;
};

