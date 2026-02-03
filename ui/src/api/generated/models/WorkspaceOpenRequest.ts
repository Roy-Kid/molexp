/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Request to open a workspace path.
 */
export type WorkspaceOpenRequest = {
    /**
     * Absolute path to the workspace
     */
    path: string;
    /**
     * Create workspace metadata if missing
     */
    create_if_missing?: boolean;
};

