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
     * @param molexpSession
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static listExperimentsApiProjectsProjectIdExperimentsGet(
        projectId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<Array<ExperimentResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments',
            path: {
                'project_id': projectId,
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
     * Create Experiment
     * @param projectId
     * @param requestBody
     * @param molexpSession
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static createExperimentApiProjectsProjectIdExperimentsPost(
        projectId: string,
        requestBody: ExperimentCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<ExperimentResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments',
            path: {
                'project_id': projectId,
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
     * Delete Experiment
     * @param projectId
     * @param experimentId
     * @param molexpSession
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteExperimentApiProjectsProjectIdExperimentsExperimentIdDelete(
        projectId: string,
        experimentId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/projects/{project_id}/experiments/{experiment_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
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
     * Get Experiment
     * @param projectId
     * @param experimentId
     * @param molexpSession
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static getExperimentApiProjectsProjectIdExperimentsExperimentIdGet(
        projectId: string,
        experimentId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<ExperimentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
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
     * Get Experiment Comparison
     * Comparison matrix: parameter columns x run rows + final metric values per run.
     * @param projectId
     * @param experimentId
     * @param molexpSession
     * @returns ExperimentComparisonResponse Successful Response
     * @throws ApiError
     */
    public static getExperimentComparisonApiProjectsProjectIdExperimentsExperimentIdComparisonGet(
        projectId: string,
        experimentId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<ExperimentComparisonResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/comparison',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
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
     * List Experiments
     * @param projectId
     * @param ws
     * @param molexpSession
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static listExperimentsApiWorkspacesWsProjectsProjectIdExperimentsGet(
        projectId: string,
        ws: string,
        molexpSession?: (string | null),
    ): CancelablePromise<Array<ExperimentResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments',
            path: {
                'project_id': projectId,
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
     * Create Experiment
     * @param projectId
     * @param ws
     * @param requestBody
     * @param molexpSession
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static createExperimentApiWorkspacesWsProjectsProjectIdExperimentsPost(
        projectId: string,
        ws: string,
        requestBody: ExperimentCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<ExperimentResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments',
            path: {
                'project_id': projectId,
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
     * Delete Experiment
     * @param projectId
     * @param experimentId
     * @param ws
     * @param molexpSession
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteExperimentApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdDelete(
        projectId: string,
        experimentId: string,
        ws: string,
        molexpSession?: (string | null),
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
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
     * Get Experiment
     * @param projectId
     * @param experimentId
     * @param ws
     * @param molexpSession
     * @returns ExperimentResponse Successful Response
     * @throws ApiError
     */
    public static getExperimentApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdGet(
        projectId: string,
        experimentId: string,
        ws: string,
        molexpSession?: (string | null),
    ): CancelablePromise<ExperimentResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
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
     * Get Experiment Comparison
     * Comparison matrix: parameter columns x run rows + final metric values per run.
     * @param projectId
     * @param experimentId
     * @param ws
     * @param molexpSession
     * @returns ExperimentComparisonResponse Successful Response
     * @throws ApiError
     */
    public static getExperimentComparisonApiWorkspacesWsProjectsProjectIdExperimentsExperimentIdComparisonGet(
        projectId: string,
        experimentId: string,
        ws: string,
        molexpSession?: (string | null),
    ): CancelablePromise<ExperimentComparisonResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/experiments/{experiment_id}/comparison',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
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
