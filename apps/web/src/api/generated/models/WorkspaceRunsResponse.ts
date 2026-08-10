/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { WorkspaceRunRow } from './WorkspaceRunRow';
import type { WorkspaceRunsStats } from './WorkspaceRunsStats';
export type WorkspaceRunsResponse = {
    runs: Array<WorkspaceRunRow>;
    stats: WorkspaceRunsStats;
    total: number;
    truncated?: boolean;
};

