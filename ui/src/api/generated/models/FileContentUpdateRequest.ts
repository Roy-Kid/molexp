/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Request to update file content.
 */
export type FileContentUpdateRequest = {
    /**
     * Workspace folder ID or 'workspace'
     */
    folder_id: string;
    /**
     * Relative path within the folder
     */
    path: string;
    /**
     * New file content
     */
    content: string;
};

