/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One background curate task's current state (UI polls this).
 */
export type CurateTaskResponse = {
    capabilityId?: (string | null);
    createdAt: string;
    error?: (string | null);
    experimentId: string;
    granted?: (boolean | null);
    model: string;
    mutationSummary?: (string | null);
    projectId: string;
    requestPreview: string;
    runId: string;
    status: string;
    taskId: string;
};

