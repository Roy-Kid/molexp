/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One background plan task's current state (UI polls this).
 */
export type PlanTaskResponse = {
    createdAt: string;
    draftPreview: string;
    error?: (string | null);
    execute?: boolean;
    experimentId: string;
    model: string;
    projectId: string;
    recordErrors?: Array<string>;
    runId: string;
    status: string;
    taskId: string;
    workflowPersisted?: boolean;
};

