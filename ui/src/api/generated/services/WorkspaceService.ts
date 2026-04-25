/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { DirectoryCreateRequest } from '../models/DirectoryCreateRequest';
import type { FileContentResponse } from '../models/FileContentResponse';
import type { FileContentUpdateRequest } from '../models/FileContentUpdateRequest';
import type { WorkspaceInfoResponse } from '../models/WorkspaceInfoResponse';
import type { WorkspaceOpenRequest } from '../models/WorkspaceOpenRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class WorkspaceService {
    /**
     * Get Workspace Info
     * Get workspace information.
     * @returns WorkspaceInfoResponse Successful Response
     * @throws ApiError
     */
    public static getWorkspaceInfoApiWorkspaceInfoGet(): CancelablePromise<WorkspaceInfoResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/info',
        });
    }
    /**
     * List Workspace Files
     * Return a nested file tree rooted at the requested path.
     *
     * With ``include=catalog``, file nodes that match a registered asset
     * are enriched with ``assetId``, ``assetKind``, ``producerRunId`` and
     * ``producerTaskId`` so the UI can render lineage chips inline.
     * @param path Workspace-relative path to list
     * @param maxDepth Maximum recursion depth
     * @param include Comma-separated optional enrichments (e.g. 'catalog')
     * @returns any Successful Response
     * @throws ApiError
     */
    public static listWorkspaceFilesApiWorkspaceFilesGet(
        path: string = '',
        maxDepth: number = 4,
        include?: (string | null),
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/files',
            query: {
                'path': path,
                'max_depth': maxDepth,
                'include': include,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Write File
     * Create or update a file in the workspace.
     * @param requestBody
     * @returns any Successful Response
     * @throws ApiError
     */
    public static writeFileApiWorkspaceFilesPut(
        requestBody: FileContentUpdateRequest,
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/workspace/files',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Read Workspace File
     * Read a text file from the workspace.
     * @param path Workspace-relative path to read
     * @returns FileContentResponse Successful Response
     * @throws ApiError
     */
    public static readWorkspaceFileApiWorkspaceFileGet(
        path: string = '',
    ): CancelablePromise<FileContentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/file',
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Read Workspace File Blob
     * Read a binary file from the workspace.
     * @param path Workspace-relative path to read
     * @returns any Successful Response
     * @throws ApiError
     */
    public static readWorkspaceFileBlobApiWorkspaceFileBlobGet(
        path: string = '',
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/file/blob',
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Open Workspace
     * Set the active workspace path.
     * @param requestBody
     * @returns WorkspaceInfoResponse Successful Response
     * @throws ApiError
     */
    public static openWorkspaceApiWorkspaceOpenPost(
        requestBody: WorkspaceOpenRequest,
    ): CancelablePromise<WorkspaceInfoResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/open',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Directory
     * Create a directory in the workspace.
     * @param requestBody
     * @returns any Successful Response
     * @throws ApiError
     */
    public static createDirectoryApiWorkspaceDirectoriesPost(
        requestBody: DirectoryCreateRequest,
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/directories',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
