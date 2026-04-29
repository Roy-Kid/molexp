/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ExecutionRecordResponse } from './ExecutionRecordResponse';
import type { WorkflowSnapshotResponse } from './WorkflowSnapshotResponse';
export type RunResponse = {
    id: string;
    projectId: string;
    experimentId: string;
    status: string;
    created: string;
    finished?: (string | null);
    parameters?: Record<string, any>;
    results?: Record<string, any>;
    workflow?: (WorkflowSnapshotResponse | null);
    workflowSource?: (string | null);
    error?: (Record<string, string> | null);
    executorInfo?: Record<string, any>;
    profile?: (string | null);
    config?: Record<string, any>;
    configHash?: (string | null);
    executionHistory?: Array<ExecutionRecordResponse>;
};

