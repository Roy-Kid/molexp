/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ExperimentComparisonResponse } from '../models/ExperimentComparisonResponse';
import type { ExperimentCreateRequest } from '../models/ExperimentCreateRequest';
import type { ExperimentResponse } from '../models/ExperimentResponse';
import type { MessageResponse } from '../models/MessageResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ExperimentsService {
    /**
     * List Experiments
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
     * Delete Experiment
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
    /**
     * Get Experiment
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
     * Get Experiment Comparison
     * Comparison matrix: parameter columns x run rows + final metric values per run.
     * @param projectId
     * @param experimentId
     * @returns ExperimentComparisonResponse Successful Response
     * @throws ApiError
     */
    public static getExperimentComparisonApiProjectsProjectIdExperimentsExperimentIdComparisonGet(
        projectId: string,
        experimentId: string,
    ): CancelablePromise<ExperimentComparisonResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/comparison',
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
     * List Experiments
     * @param projectId
     * @param ws
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static listExperimentsApiWorkspacesWsProjectsProjectIdExperimentsGet(
        projectId: string,
        ws: string,
    ): CancelablePromise<Array<ExperimentResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments',
            path: {
                'project_id': projectId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Experiment
     * @param projectId
     * @param ws
     * @param requestBody
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static createExperimentApiWorkspacesWsProjectsProjectIdExperimentsPost(
        projectId: string,
        ws: string,
        requestBody: ExperimentCreateRequest,
    ): CancelablePromise<ExperimentResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments',
            path: {
                'project_id': projectId,
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
     * Delete Experiment
     * @param projectId
     * @param experimentId
     * @param ws
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteExperimentApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdDelete(
        projectId: string,
        experimentId: string,
        ws: string,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Experiment
     * @param projectId
     * @param experimentId
     * @param ws
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static getExperimentApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdGet(
        projectId: string,
        experimentId: string,
        ws: string,
    ): CancelablePromise<ExperimentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Experiment Comparison
     * Comparison matrix: parameter columns x run rows + final metric values per run.
     * @param projectId
     * @param experimentId
     * @param ws
     * @returns ExperimentComparisonResponse Successful Response
     * @throws ApiError
     */
    public static getExperimentComparisonApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdComparisonGet(
        projectId: string,
        experimentId: string,
        ws: string,
    ): CancelablePromise<ExperimentComparisonResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/comparison',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
