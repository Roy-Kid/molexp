/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ExecutionRecordResponse } from './ExecutionRecordResponse';
import type { WorkflowSnapshotResponse } from './WorkflowSnapshotResponse';
export type RunResponse = {
    config?: Record<string, any>;
    configHash?: (string | null);
    created: string;
    error?: (Record<string, string> | null);
    executionHistory?: Array<ExecutionRecordResponse>;
    executorInfo?: Record<string, any>;
    experimentId: string;
    finished?: (string | null);
    id: string;
    parameters?: Record<string, any>;
    profile?: (string | null);
    projectId: string;
    results?: Record<string, any>;
    status: string;
    target?: (string | null);
    workflow?: (WorkflowSnapshotResponse | null);
    workflowSource?: (string | null);
};

