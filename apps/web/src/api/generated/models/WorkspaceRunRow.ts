/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { JSONValue } from './JSONValue';
import type { WorkspaceExecutionRow } from './WorkspaceExecutionRow';
/**
 * One run, with its execution history nested for tree expansion.
 */
export type WorkspaceRunRow = {
    backend?: (string | null);
    cluster?: (string | null);
    createdAt: string;
    executionCount?: number;
    executions?: Array<WorkspaceExecutionRow>;
    experimentId: string;
    experimentName: string;
    finishedAt?: (string | null);
    id: string;
    latestSchedulerJobId?: (string | null);
    name: string;
    parameters?: Record<string, JSONValue>;
    profile?: (string | null);
    projectId: string;
    projectName: string;
    scheduler?: (string | null);
    status: string;
    target?: (string | null);
};

