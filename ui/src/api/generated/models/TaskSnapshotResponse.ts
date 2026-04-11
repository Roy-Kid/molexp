/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Task snapshot response.
 */
export type TaskSnapshotResponse = {
    taskId: string;
    taskType: string;
    codeHash: string;
    configHash: string;
    codeSource?: string;
    snapshotKey: string;
    createdAt: string;
    configData?: Record<string, any>;
};
