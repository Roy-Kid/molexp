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
     * @returns CacheClearResponse Successful Response
     * @throws ApiError
     */
    public static clearCacheApiCacheDelete(): CancelablePromise<CacheClearResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/cache',
        });
    }
    /**
     * Get Cache Stats
     * @returns CacheStatsResponse Successful Response
     * @throws ApiError
     */
    public static getCacheStatsApiCacheStatsGet(): CancelablePromise<CacheStatsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/cache/stats',
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
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static createExecutionApiExecutionsPost(
        requestBody: ExecutionCreateRequest,
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/executions',
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
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getExecutionPlanApiPlanPost(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/plan',
        });
    }
}
