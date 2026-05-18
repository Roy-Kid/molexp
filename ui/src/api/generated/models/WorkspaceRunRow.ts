/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { WorkspaceExecutionRow } from './WorkspaceExecutionRow';
/**
 * One run, with its execution history nested for tree expansion.
 */
export type WorkspaceRunRow = {
    id: string;
    name: string;
    projectId: string;
    projectName: string;
    experimentId: string;
    experimentName: string;
    status: string;
    backend?: (string | null);
    cluster?: (string | null);
    scheduler?: (string | null);
    target?: (string | null);
    profile?: (string | null);
    parameters?: Record<string, any>;
    createdAt: string;
    finishedAt?: (string | null);
    executionCount?: number;
    latestSchedulerJobId?: (string | null);
    executions?: Array<WorkspaceExecutionRow>;
};
