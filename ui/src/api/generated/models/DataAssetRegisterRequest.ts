/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Register an existing workspace file in place as a ``DataAsset``.
 */
export type DataAssetRegisterRequest = {
    /**
     * Free-form tags
     */
    metadata?: Record<string, string>;
    /**
     * Display name (defaults to the file name)
     */
    name?: (string | null);
    /**
     * Workspace-relative path to the existing file
     */
    path: string;
};

