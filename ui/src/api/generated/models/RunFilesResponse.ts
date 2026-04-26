/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunFileNode } from './RunFileNode';
/**
 * Per-run output file tree, enriched with catalog producer metadata.
 */
export type RunFilesResponse = {
    runId: string;
    runDir: string;
    nodes?: Array<RunFileNode>;
};
