/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TaskSnapshotResponse } from './TaskSnapshotResponse';
/**
 * Execution plan response.
 */
export type ExecutionPlanResponse = {
    plan: Array<string>;
    nodeCount: number;
    snapshots?: Array<TaskSnapshotResponse>;
};
