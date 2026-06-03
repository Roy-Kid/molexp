/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { MolqJobDetailResponse } from '../models/MolqJobDetailResponse';
import type { MolqJobsResponse } from '../models/MolqJobsResponse';
import type { MolqTargetListResponse } from '../models/MolqTargetListResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class MolqService {
    /**
     * List Jobs
     * List jobs across one or all targets, plus aggregate queue stats.
     * @param target Profile name to filter by.
     * @param includeTerminal
     * @param limit
     * @returns MolqJobsResponse Successful Response
     * @throws ApiError
     */
    public static listJobsApiPluginsMolqJobsGet(
        target?: (string | null),
        includeTerminal: boolean = true,
        limit: number = 200,
    ): CancelablePromise<MolqJobsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins/molq/jobs',
            query: {
                'target': target,
                'includeTerminal': includeTerminal,
                'limit': limit,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Job
     * Return a single job's detail including transitions and dependency state.
     * @param jobId
     * @param target Profile name owning the job.
     * @returns MolqJobDetailResponse Successful Response
     * @throws ApiError
     */
    public static getJobApiPluginsMolqJobsJobIdGet(
        jobId: string,
        target: string,
    ): CancelablePromise<MolqJobDetailResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins/molq/jobs/{job_id}',
            path: {
                'job_id': jobId,
            },
            query: {
                'target': target,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Stream Logs
     * SSE stream of newline-terminated log chunks.
     *
     * Each event payload is ``data: {"line": "..."}\n\n`` so the client's
     * EventSource ``message`` handler parses one log line per event.
     * @param jobId
     * @param target Profile name owning the job.
     * @param stream
     * @returns any Successful Response
     * @throws ApiError
     */
    public static streamLogsApiPluginsMolqJobsJobIdLogsGet(
        jobId: string,
        target: string,
        stream: string = 'stdout',
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins/molq/jobs/{job_id}/logs',
            path: {
                'job_id': jobId,
            },
            query: {
                'target': target,
                'stream': stream,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Targets
     * List configured molq targets (one per profile in ``~/.molq/config.toml``).
     * @returns MolqTargetListResponse Successful Response
     * @throws ApiError
     */
    public static listTargetsApiPluginsMolqTargetsGet(): CancelablePromise<MolqTargetListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins/molq/targets',
        });
    }
}
