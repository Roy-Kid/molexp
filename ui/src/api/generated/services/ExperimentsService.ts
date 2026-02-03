/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ExperimentCreateRequest } from '../models/ExperimentCreateRequest';
import type { ExperimentResponse } from '../models/ExperimentResponse';
import type { MessageResponse } from '../models/MessageResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ExperimentsService {
    /**
     * List Experiments
     * List experiments in a project.
     * @param projectId
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static listExperimentsApiProjectsProjectIdExperimentsGet(
        projectId: string,
    ): CancelablePromise<Array<ExperimentResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments',
            path: {
                'project_id': projectId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Experiment
     * Create a new experiment.
     * @param projectId
     * @param requestBody
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static createExperimentApiProjectsProjectIdExperimentsPost(
        projectId: string,
        requestBody: ExperimentCreateRequest,
    ): CancelablePromise<ExperimentResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments',
            path: {
                'project_id': projectId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Experiment
     * Get experiment details.
     * @param projectId
     * @param experimentId
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static getExperimentApiProjectsProjectIdExperimentsExperimentIdGet(
        projectId: string,
        experimentId: string,
    ): CancelablePromise<ExperimentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}',
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
     * Delete Experiment
     * Delete an experiment.
     * @param projectId
     * @param experimentId
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteExperimentApiProjectsProjectIdExperimentsExperimentIdDelete(
        projectId: string,
        experimentId: string,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/projects/{project_id}/experiments/{experiment_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
