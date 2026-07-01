/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { GitCheckpointResponse } from '../models/GitCheckpointResponse';
import type { GitPushRequest } from '../models/GitPushRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class GitService {
    /**
     * Git Checkpoint Route
     * @returns GitCheckpointResponse Successful Response
     * @throws ApiError
     */
    public static gitCheckpointRouteApiGitCheckpointPost(): CancelablePromise<GitCheckpointResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/git/checkpoint',
        });
    }
    /**
     * Git Push Route
     * @param requestBody
     * @returns string Successful Response
     * @throws ApiError
     */
    public static gitPushRouteApiGitPushPost(
        requestBody: GitPushRequest,
    ): CancelablePromise<Record<string, string>> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/git/push',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Git Rebuild Route
     * @returns GitCheckpointResponse Successful Response
     * @throws ApiError
     */
    public static gitRebuildRouteApiGitRebuildPost(): CancelablePromise<GitCheckpointResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/git/rebuild',
        });
    }
    /**
     * Git Checkpoint Route
     * @param ws
     * @returns GitCheckpointResponse Successful Response
     * @throws ApiError
     */
    public static gitCheckpointRouteApiWorkspacesWsGitCheckpointPost(
        ws: string,
    ): CancelablePromise<GitCheckpointResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/git/checkpoint',
            path: {
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Git Push Route
     * @param ws
     * @param requestBody
     * @returns string Successful Response
     * @throws ApiError
     */
    public static gitPushRouteApiWorkspacesWsGitPushPost(
        ws: string,
        requestBody: GitPushRequest,
    ): CancelablePromise<Record<string, string>> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/git/push',
            path: {
                'ws': ws,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Git Rebuild Route
     * @param ws
     * @returns GitCheckpointResponse Successful Response
     * @throws ApiError
     */
    public static gitRebuildRouteApiWorkspacesWsGitRebuildPost(
        ws: string,
    ): CancelablePromise<GitCheckpointResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/git/rebuild',
            path: {
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
