/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RunCreateRequest } from '../models/RunCreateRequest';
import type { RunResponse } from '../models/RunResponse';
import type { RunStatusResponse } from '../models/RunStatusResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class RunsService {
    /**
     * List Runs
     * List runs in an experiment.
     * @param projectId
     * @param experimentId
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static listRunsApiProjectsProjectIdExperimentsExperimentIdRunsGet(
        projectId: string,
        experimentId: string,
    ): CancelablePromise<Array<RunResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Run
     * Create a new run.
     * @param projectId
     * @param experimentId
     * @param requestBody
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static createRunApiProjectsProjectIdExperimentsExperimentIdRunsPost(
        projectId: string,
        experimentId: string,
        requestBody: RunCreateRequest,
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Run
     * Get run details.
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunResponse Successful Response
     * @throws ApiError
     */
    public static getRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdGet(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Update Run Status
     * Update run status.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param requestBody
     * @returns RunStatusResponse Successful Response
     * @throws ApiError
     */
    public static updateRunStatusApiProjectsProjectIdExperimentsExperimentIdRunsRunIdStatusPatch(
        projectId: string,
        experimentId: string,
        runId: string,
        requestBody: Record<string, string>,
    ): CancelablePromise<RunStatusResponse> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/status',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Start Run
     * Start run execution.
     * @param projectId
     * @param experimentId
     * @param runId
     * @returns RunStatusResponse Successful Response
     * @throws ApiError
     */
    public static startRunApiProjectsProjectIdExperimentsExperimentIdRunsRunIdStartPost(
        projectId: string,
        experimentId: string,
        runId: string,
    ): CancelablePromise<RunStatusResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/start',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
