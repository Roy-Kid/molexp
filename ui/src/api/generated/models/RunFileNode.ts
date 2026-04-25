/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One node in a run's output file tree.
 */
export type RunFileNode = {
    name: string;
    relPath: string;
    type: string;
    size?: (number | null);
    modified?: (number | null);
    assetId?: (string | null);
    assetKind?: (string | null);
    taskId?: (string | null);
    children?: Array<RunFileNode>;
};

