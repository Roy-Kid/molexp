/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ServedWorkspaceResponse } from '../models/ServedWorkspaceResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class WorkspacesService {
    /**
     * List Workspaces
     * List the workspaces ``molexp serve`` was started with.
     *
     * A remote workspace whose transport is currently unreachable is still
     * listed, flagged ``unreachable`` so the UI can degrade gracefully rather
     * than failing the whole list.
     *
     * When auth is enabled, the list is filtered by the user's workspace allowlist.
     * @param molexpSession
     * @returns ServedWorkspaceResponse Successful Response
     * @throws ApiError
     */
    public static listWorkspacesApiWorkspacesGet(
        molexpSession?: (string | null),
    ): CancelablePromise<Array<ServedWorkspaceResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
