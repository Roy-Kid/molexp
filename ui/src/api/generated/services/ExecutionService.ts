/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CacheClearResponse } from '../models/CacheClearResponse';
import type { CacheStatsResponse } from '../models/CacheStatsResponse';
import type { ExecutionCreateRequest } from '../models/ExecutionCreateRequest';
import type { ExecutionPlanRequest } from '../models/ExecutionPlanRequest';
import type { ExecutionPlanResponse } from '../models/ExecutionPlanResponse';
import type { RunResponse } from '../models/RunResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ExecutionService {
    /**
     * Create Execution
     * Create a new execution in a specific project/experiment.
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
     * Get execution plan for a workflow definition.
     * @param requestBody
     * @returns ExecutionPlanResponse Successful Response
     * @throws ApiError
     */
    public static getExecutionPlanApiPlanPost(
        requestBody: ExecutionPlanRequest,
    ): CancelablePromise<ExecutionPlanResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/plan',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Cache Stats
     * Get cache statistics.
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
     * Clear Cache
     * Clear all cache entries.
     * @returns CacheClearResponse Successful Response
     * @throws ApiError
     */
    public static clearCacheApiCacheDelete(): CancelablePromise<CacheClearResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/cache',
        });
    }
}
