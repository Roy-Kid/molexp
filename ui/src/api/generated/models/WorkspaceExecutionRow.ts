/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One execution attempt of a run, surfaced for the workspace runs table.
 */
export type WorkspaceExecutionRow = {
    backend?: (string | null);
    backendMetadata?: Record<string, string>;
    durationSeconds?: (number | null);
    executionId: string;
    finishedAt?: (string | null);
    runId: string;
    schedulerJobId?: (string | null);
    startedAt: string;
    status: string;
};

