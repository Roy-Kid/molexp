/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TensorboardScalarsResponse } from '../models/TensorboardScalarsResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class TensorboardService {
    /**
     * Get Run Tensorboard Scalars
     * Parse all (or filtered) scalar tags from a run's tfevents files.
     *
     * Returns 503 with an install hint when the optional ``tensorboard``
     * extra is missing, 404 when the run is unknown, and 200 with an
     * empty ``series`` list when the run has no tfevents on disk.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param tag Repeatable scalar-tag filter
     * @param logdir Relative path under run_dir; default = discover every tfevents dir
     * @returns TensorboardScalarsResponse Successful Response
     * @throws ApiError
     */
    public static getRunTensorboardScalarsApiProjectsProjectIdExperimentsExperimentIdRunsRunIdTensorboardScalarsGet(
        projectId: string,
        experimentId: string,
        runId: string,
        tag?: (Array<string> | null),
        logdir?: (string | null),
    ): CancelablePromise<TensorboardScalarsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/runs/{run_id}/tensorboard/scalars',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
            },
            query: {
                'tag': tag,
                'logdir': logdir,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
