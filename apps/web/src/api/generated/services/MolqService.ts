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
     * @param molexpSession
     * @returns MolqJobsResponse Successful Response
     * @throws ApiError
     */
    public static listJobsApiPluginsMolqJobsGet(
        target?: (string | null),
        includeTerminal: boolean = true,
        limit: number = 200,
        molexpSession?: (string | null),
    ): CancelablePromise<MolqJobsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins/molq/jobs',
            cookies: {
                'molexp_session': molexpSession,
            },
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
     * @param molexpSession
     * @returns MolqJobDetailResponse Successful Response
     * @throws ApiError
     */
    public static getJobApiPluginsMolqJobsJobIdGet(
        jobId: string,
        target: string,
        molexpSession?: (string | null),
    ): CancelablePromise<MolqJobDetailResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins/molq/jobs/{job_id}',
            path: {
                'job_id': jobId,
            },
            cookies: {
                'molexp_session': molexpSession,
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
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static streamLogsApiPluginsMolqJobsJobIdLogsGet(
        jobId: string,
        target: string,
        stream: string = 'stdout',
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins/molq/jobs/{job_id}/logs',
            path: {
                'job_id': jobId,
            },
            cookies: {
                'molexp_session': molexpSession,
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
     * List configured molq targets (one per profile in ``~/.molq/config.yaml``).
     * @param molexpSession
     * @returns MolqTargetListResponse Successful Response
     * @throws ApiError
     */
    public static listTargetsApiPluginsMolqTargetsGet(
        molexpSession?: (string | null),
    ): CancelablePromise<MolqTargetListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins/molq/targets',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
