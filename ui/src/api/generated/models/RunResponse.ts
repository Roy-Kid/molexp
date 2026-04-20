/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { WorkflowSnapshotResponse } from './WorkflowSnapshotResponse';
export type RunResponse = {
    id: string;
    projectId: string;
    experimentId: string;
    status: string;
    created: string;
    finished?: (string | null);
    parameters?: Record<string, any>;
    workflow?: (WorkflowSnapshotResponse | null);
    error?: (Record<string, string> | null);
    executorInfo?: Record<string, any>;
    profile?: (string | null);
    config?: Record<string, any>;
    configHash?: (string | null);
};

