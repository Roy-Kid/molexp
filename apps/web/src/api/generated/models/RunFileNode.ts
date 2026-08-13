/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One node in a run's output file tree.
 */
export type RunFileNode = {
    assetId?: (string | null);
    assetKind?: (string | null);
    children?: Array<RunFileNode>;
    modified?: (number | null);
    name: string;
    relPath: string;
    size?: (number | null);
    taskId?: (string | null);
    type: string;
};

