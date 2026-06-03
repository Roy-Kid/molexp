/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CacheControlRequest } from '../models/CacheControlRequest';
import type { CacheControlResponse } from '../models/CacheControlResponse';
import type { DirectoryCreateRequest } from '../models/DirectoryCreateRequest';
import type { FileContentResponse } from '../models/FileContentResponse';
import type { FileContentUpdateRequest } from '../models/FileContentUpdateRequest';
import type { TargetTestResponse } from '../models/TargetTestResponse';
import type { WorkspaceInfoResponse } from '../models/WorkspaceInfoResponse';
import type { WorkspaceOpenLocalRequest } from '../models/WorkspaceOpenLocalRequest';
import type { WorkspaceOpenRemoteRequest } from '../models/WorkspaceOpenRemoteRequest';
import type { WorkspaceRunsResponse } from '../models/WorkspaceRunsResponse';
import type { WorkspaceTargetCreateRequest } from '../models/WorkspaceTargetCreateRequest';
import type { WorkspaceTargetListResponse } from '../models/WorkspaceTargetListResponse';
import type { WorkspaceTargetResponse } from '../models/WorkspaceTargetResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class WorkspaceService {
    /**
     * Invalidate Workspace Cache
     * Drop cached entries from the active workspace's mirror.
     *
     * ``scope="indices"`` is the "I added a run on the remote, refresh
     * navigation" knob — it drops only entries whose basename identifies
     * a navigation-index file, leaving log/blob bytes intact.
     * @param requestBody
     * @returns CacheControlResponse Successful Response
     * @throws ApiError
     */
    public static invalidateWorkspaceCacheApiWorkspaceCacheInvalidatePost(
        requestBody: CacheControlRequest,
    ): CancelablePromise<CacheControlResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/cache/invalidate',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Refresh Workspace Cache
     * Invalidate, then walk the navigation indices again.
     *
     * Saves the UI from issuing a follow-up call after a refresh button
     * click.  Per-node failures during the walk surface as ``warnings`` —
     * the response is still 200 so a single bad project does not blank
     * the whole tree.
     * @param requestBody
     * @returns CacheControlResponse Successful Response
     * @throws ApiError
     */
    public static refreshWorkspaceCacheApiWorkspaceCacheRefreshPost(
        requestBody: CacheControlRequest,
    ): CancelablePromise<CacheControlResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/cache/refresh',
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
    /**
     * Read Workspace File
     * Read a text file from the workspace.
     *
     * Routes through ``workspace._fs`` so remote workspaces (and the
     * :class:`CachedRemoteFileSystem` mirror) take effect.
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
     *
     * Routes through ``workspace._fs`` so remote workspaces (and the
     * :class:`CachedRemoteFileSystem` mirror) take effect.
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
     * Open Workspace
     * Set the active workspace — local path or registered remote descriptor.
     *
     * Switching the active workspace drains any registered workspace
     * subscribers (SSE streams, file watchers — registered via
     * :func:`~molexp.server.dependencies.register_workspace_subscriber`)
     * *before* the cache is reset, so the new workspace starts from a
     * clean subscriber slate.
     * @param requestBody
     * @returns WorkspaceInfoResponse Successful Response
     * @throws ApiError
     */
    public static openWorkspaceApiWorkspaceOpenPost(
        requestBody: (WorkspaceOpenLocalRequest | WorkspaceOpenRemoteRequest),
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
     * List Workspace Runs
     * Cross-experiment list of runs, each with embedded execution attempts.
     *
     * Returns rows ordered by ``created_at`` desc.  Plugins surface
     * backend-specific columns (cluster, scheduler job id, etc.) via the
     * ``backend`` / ``backendMetadata`` fields on each execution row.
     * @param projectId
     * @param experimentId
     * @param backend Filter by executor backend
     * @param status Filter by run status
     * @param limit
     * @returns WorkspaceRunsResponse Successful Response
     * @throws ApiError
     */
    public static listWorkspaceRunsApiWorkspaceRunsGet(
        projectId?: (string | null),
        experimentId?: (string | null),
        backend?: (string | null),
        status?: (string | null),
        limit: number = 500,
    ): CancelablePromise<WorkspaceRunsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/runs',
            query: {
                'projectId': projectId,
                'experimentId': experimentId,
                'backend': backend,
                'status': status,
                'limit': limit,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Workspace Targets
     * @returns WorkspaceTargetListResponse Successful Response
     * @throws ApiError
     */
    public static listWorkspaceTargetsApiWorkspaceTargetsGet(): CancelablePromise<WorkspaceTargetListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/targets',
        });
    }
    /**
     * Create Workspace Target
     * @param requestBody
     * @returns WorkspaceTargetResponse Successful Response
     * @throws ApiError
     */
    public static createWorkspaceTargetApiWorkspaceTargetsPost(
        requestBody: WorkspaceTargetCreateRequest,
    ): CancelablePromise<WorkspaceTargetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/targets',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Delete Workspace Target
     * @param name
     * @returns void
     * @throws ApiError
     */
    public static deleteWorkspaceTargetApiWorkspaceTargetsNameDelete(
        name: string,
    ): CancelablePromise<void> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/workspace/targets/{name}',
            path: {
                'name': name,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Test Workspace Target
     * Connectivity probe for a workspace-target descriptor.
     *
     * Returns HTTP 200 with ``ok=False`` on probe failure (matches the
     * ``/api/targets/{name}/test`` pattern) so the UI can render failures
     * inline rather than parsing HTTP error envelopes.
     * @param name
     * @returns TargetTestResponse Successful Response
     * @throws ApiError
     */
    public static testWorkspaceTargetApiWorkspaceTargetsNameTestPost(
        name: string,
    ): CancelablePromise<TargetTestResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/targets/{name}/test',
            path: {
                'name': name,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
