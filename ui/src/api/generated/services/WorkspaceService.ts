/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CacheControlRequest } from '../models/CacheControlRequest';
import type { CacheControlResponse } from '../models/CacheControlResponse';
import type { CacheStatusResponse } from '../models/CacheStatusResponse';
import type { CurateRequest } from '../models/CurateRequest';
import type { CurateResponse } from '../models/CurateResponse';
import type { DirectoryCreateRequest } from '../models/DirectoryCreateRequest';
import type { FileContentResponse } from '../models/FileContentResponse';
import type { FileContentUpdateRequest } from '../models/FileContentUpdateRequest';
import type { TargetTestResponse } from '../models/TargetTestResponse';
import type { WorkspaceContextResponse } from '../models/WorkspaceContextResponse';
import type { WorkspaceEventResponse } from '../models/WorkspaceEventResponse';
import type { WorkspaceInfoResponse } from '../models/WorkspaceInfoResponse';
import type { WorkspaceOpenLocalRequest } from '../models/WorkspaceOpenLocalRequest';
import type { WorkspaceOpenRemoteRequest } from '../models/WorkspaceOpenRemoteRequest';
import type { WorkspaceRunsResponse } from '../models/WorkspaceRunsResponse';
import type { WorkspaceSummaryResponse } from '../models/WorkspaceSummaryResponse';
import type { WorkspaceTargetCreateRequest } from '../models/WorkspaceTargetCreateRequest';
import type { WorkspaceTargetListResponse } from '../models/WorkspaceTargetListResponse';
import type { WorkspaceTargetResponse } from '../models/WorkspaceTargetResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class WorkspaceService {
    /**
     * Get Workspace Events
     * The workspace-wide activity stream, newest first.
     *
     * The global read over the event spine — the same shared
     * :func:`molexp.workspace.events.read_workspace_events` code path the
     * per-run route and ``molexp runs info`` use. A workspace with no timeline
     * yet answers ``[]`` without creating the DB (reading is side-effect free).
     * @param type Keep only this event type
     * @param ref Keep only events referencing this id
     * @param limit
     * @param molexpSession
     * @returns WorkspaceEventResponse Successful Response
     * @throws ApiError
     */
    public static getWorkspaceEventsApiEventsGet(
        type?: ('run.created' | 'run.started' | 'run.failed' | 'run.completed' | 'asset.added' | 'knowledge.created' | 'workflow.created' | 'experiment.created' | null),
        ref?: (string | null),
        limit: number = 50,
        molexpSession?: (string | null),
    ): CancelablePromise<Array<WorkspaceEventResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/events',
            cookies: {
                'molexp_session': molexpSession,
            },
            query: {
                'type': type,
                'ref': ref,
                'limit': limit,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Invalidate Workspace Cache
     * Drop cached entries from the active workspace's mirror.
     *
     * ``scope="indices"`` is the "I added a run on the remote, refresh
     * navigation" knob — it drops only entries whose basename identifies
     * a navigation-index file, leaving log/blob bytes intact.
     * @param requestBody
     * @param molexpSession
     * @returns CacheControlResponse Successful Response
     * @throws ApiError
     */
    public static invalidateWorkspaceCacheApiWorkspaceCacheInvalidatePost(
        requestBody: CacheControlRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<CacheControlResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/cache/invalidate',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns CacheControlResponse Successful Response
     * @throws ApiError
     */
    public static refreshWorkspaceCacheApiWorkspaceCacheRefreshPost(
        requestBody: CacheControlRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<CacheControlResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/cache/refresh',
            cookies: {
                'molexp_session': molexpSession,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Workspace Cache Status
     * Poll remote-index progress (file-count total → fetch done).
     *
     * Local workspaces return ``cached=false`` with idle progress. The UI
     * status strip polls this while ``phase`` is ``counting`` / ``fetching``.
     * @param molexpSession
     * @returns CacheStatusResponse Successful Response
     * @throws ApiError
     */
    public static workspaceCacheStatusApiWorkspaceCacheStatusGet(
        molexpSession?: (string | null),
    ): CancelablePromise<CacheStatusResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/cache/status',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Workspace Context
     * The canonical structural workspace read-model (integration.md §1).
     *
     * A read-only projection assembled from authoritative workspace state — the one
     * shape agents/planners/CLI/UI observe. ``ContextFocus`` is supplied by the
     * caller via optional query params and is never persisted. ``/runs`` remains the
     * specialized detailed run view (richer per-execution rows); this endpoint is the
     * canonical *structure* and stays consistent with it.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param molexpSession
     * @returns WorkspaceContextResponse Successful Response
     * @throws ApiError
     */
    public static getWorkspaceContextApiWorkspaceContextGet(
        projectId?: (string | null),
        experimentId?: (string | null),
        runId?: (string | null),
        molexpSession?: (string | null),
    ): CancelablePromise<WorkspaceContextResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/context',
            cookies: {
                'molexp_session': molexpSession,
            },
            query: {
                'projectId': projectId,
                'experimentId': experimentId,
                'runId': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Workspace Copilot
     * The read-only Workspace Copilot summary — structured state + ranked next-actions.
     *
     * A pure projection over the canonical ``WorkspaceContext``; it mutates nothing.
     * Next-actions are **advisory** and separated from execution — high-risk ones are
     * flagged ``requiresProposal`` (they must go through a ``ChangeProposal`` first).
     * @param molexpSession
     * @returns WorkspaceSummaryResponse Successful Response
     * @throws ApiError
     */
    public static getWorkspaceCopilotApiWorkspaceCopilotGet(
        molexpSession?: (string | null),
    ): CancelablePromise<WorkspaceSummaryResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/copilot',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Curate Workspace
     * Gate + execute one deterministic destructive-curation op (single stack).
     *
     * Shares the ``run_curation_proposal`` backend with ``molexp curate`` (Python ≡
     * UI). ``approve=false`` (default) records the proposal and refuses; ``true``
     * executes the mutation. Either way the §8 ``change_proposal`` artifact is the audit.
     * @param requestBody
     * @param molexpSession
     * @returns CurateResponse Successful Response
     * @throws ApiError
     */
    public static curateWorkspaceApiWorkspaceCuratePost(
        requestBody: CurateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<CurateResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/curate',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static createDirectoryApiWorkspaceDirectoriesPost(
        requestBody: DirectoryCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/directories',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns FileContentResponse Successful Response
     * @throws ApiError
     */
    public static readWorkspaceFileApiWorkspaceFileGet(
        path: string = '',
        molexpSession?: (string | null),
    ): CancelablePromise<FileContentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/file',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static readWorkspaceFileBlobApiWorkspaceFileBlobGet(
        path: string = '',
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/file/blob',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * Routes through ``workspace._fs`` so remote workspaces (and the
     * :class:`CachedRemoteFileSystem` mirror) work the same as local ones.
     *
     * With ``include=catalog``, file nodes that match a registered asset
     * are enriched with ``assetId``, ``assetKind``, ``producerRunId`` and
     * ``producerTaskId`` so the UI can render lineage chips inline.
     *
     * Children matching the workspace ``.gitignore`` cascade (plus a safety
     * floor for ``node_modules`` / ``.git`` / venvs) are omitted so git-managed
     * workspaces do not dump dependency trees into the UI.
     * @param path Workspace-relative path to list
     * @param maxDepth Maximum recursion depth
     * @param include Comma-separated optional enrichments (e.g. 'catalog')
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static listWorkspaceFilesApiWorkspaceFilesGet(
        path: string = '',
        maxDepth: number = 4,
        include?: (string | null),
        molexpSession?: (string | null),
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/files',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static writeFileApiWorkspaceFilesPut(
        requestBody: FileContentUpdateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<Record<string, any>> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/workspace/files',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns WorkspaceInfoResponse Successful Response
     * @throws ApiError
     */
    public static getWorkspaceInfoApiWorkspaceInfoGet(
        molexpSession?: (string | null),
    ): CancelablePromise<WorkspaceInfoResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/info',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
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
     * @param molexpSession
     * @returns WorkspaceInfoResponse Successful Response
     * @throws ApiError
     */
    public static openWorkspaceApiWorkspaceOpenPost(
        requestBody: (WorkspaceOpenLocalRequest | WorkspaceOpenRemoteRequest),
        molexpSession?: (string | null),
    ): CancelablePromise<WorkspaceInfoResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/open',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns WorkspaceRunsResponse Successful Response
     * @throws ApiError
     */
    public static listWorkspaceRunsApiWorkspaceRunsGet(
        projectId?: (string | null),
        experimentId?: (string | null),
        backend?: (string | null),
        status?: (string | null),
        limit: number = 500,
        molexpSession?: (string | null),
    ): CancelablePromise<WorkspaceRunsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/runs',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns WorkspaceTargetListResponse Successful Response
     * @throws ApiError
     */
    public static listWorkspaceTargetsApiWorkspaceTargetsGet(
        molexpSession?: (string | null),
    ): CancelablePromise<WorkspaceTargetListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspace/targets',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Workspace Target
     * @param requestBody
     * @param molexpSession
     * @returns WorkspaceTargetResponse Successful Response
     * @throws ApiError
     */
    public static createWorkspaceTargetApiWorkspaceTargetsPost(
        requestBody: WorkspaceTargetCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<WorkspaceTargetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/targets',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns void
     * @throws ApiError
     */
    public static deleteWorkspaceTargetApiWorkspaceTargetsNameDelete(
        name: string,
        molexpSession?: (string | null),
    ): CancelablePromise<void> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/workspace/targets/{name}',
            path: {
                'name': name,
            },
            cookies: {
                'molexp_session': molexpSession,
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
     * @param molexpSession
     * @returns TargetTestResponse Successful Response
     * @throws ApiError
     */
    public static testWorkspaceTargetApiWorkspaceTargetsNameTestPost(
        name: string,
        molexpSession?: (string | null),
    ): CancelablePromise<TargetTestResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspace/targets/{name}/test',
            path: {
                'name': name,
            },
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
