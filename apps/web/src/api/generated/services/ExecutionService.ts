/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CacheClearResponse } from '../models/CacheClearResponse';
import type { CacheStatsResponse } from '../models/CacheStatsResponse';
import type { ExecutionCreateRequest } from '../models/ExecutionCreateRequest';
import type { RunResponse } from '../models/RunResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ExecutionService {
    /**
     * Clear Cache
     * @param molexpSession
     * @returns CacheClearResponse Successful Response
     * @throws ApiError
     */
    public static clearCacheApiCacheDelete(
        molexpSession?: (string | null),
    ): CancelablePromise<CacheClearResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/cache',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Cache Stats
     * @param molexpSession
     * @returns CacheStatsResponse Successful Response
     * @throws ApiError
     */
    public static getCacheStatsApiCacheStatsGet(
        molexpSession?: (string | null),
    ): CancelablePromise<CacheStatsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/cache/stats',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Execution
     * Create a new execution in a specific project/experiment.
     *
     * If ``request.workflow_json`` is supplied and the experiment has no
     * workflow bound, compile and persist the IR before the run is
     * materialized so worker processes can pick it up off disk.
     * @param requestBody
     * @param molexpSession
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static createExecutionApiExecutionsPost(
        requestBody: ExecutionCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/executions',
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
     * Get Execution Plan
     * Get execution plan for a workflow (not yet implemented).
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getExecutionPlanApiPlanPost(
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/plan',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Clear Cache
     * @param ws
     * @param molexpSession
     * @returns CacheClearResponse Successful Response
     * @throws ApiError
     */
    public static clearCacheApiWorkspacesWsCacheDelete(
        ws: string,
        molexpSession?: (string | null),
    ): CancelablePromise<CacheClearResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/workspaces/{ws}/cache',
            path: {
                'ws': ws,
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
     * Get Cache Stats
     * @param ws
     * @param molexpSession
     * @returns CacheStatsResponse Successful Response
     * @throws ApiError
     */
    public static getCacheStatsApiWorkspacesWsCacheStatsGet(
        ws: string,
        molexpSession?: (string | null),
    ): CancelablePromise<CacheStatsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/cache/stats',
            path: {
                'ws': ws,
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
     * Create Execution
     * Create a new execution in a specific project/experiment.
     *
     * If ``request.workflow_json`` is supplied and the experiment has no
     * workflow bound, compile and persist the IR before the run is
     * materialized so worker processes can pick it up off disk.
     * @param ws
     * @param requestBody
     * @param molexpSession
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static createExecutionApiWorkspacesWsExecutionsPost(
        ws: string,
        requestBody: ExecutionCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/executions',
            path: {
                'ws': ws,
            },
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
     * Get Execution Plan
     * Get execution plan for a workflow (not yet implemented).
     * @param ws
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getExecutionPlanApiWorkspacesWsPlanPost(
        ws: string,
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/plan',
            path: {
                'ws': ws,
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
