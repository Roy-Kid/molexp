/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { PlanDetailResponse } from '../models/PlanDetailResponse';
import type { PlanListResponse } from '../models/PlanListResponse';
import type { WorkspacePlanListResponse } from '../models/WorkspacePlanListResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class PlansService {
    /**
     * List All Plans
     * List every generated plan in the active workspace (across all experiments).
     * @param molexpSession
     * @returns WorkspacePlanListResponse Successful Response
     * @throws ApiError
     */
    public static listAllPlansApiPlansGet(
        molexpSession?: (string | null),
    ): CancelablePromise<WorkspacePlanListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plans',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Plans
     * List the experiment's runs that carry a generated plan (experiment_report).
     * @param projectId
     * @param experimentId
     * @param molexpSession
     * @returns PlanListResponse Successful Response
     * @throws ApiError
     */
    public static listPlansApiProjectsProjectIdExperimentsExperimentIdPlansGet(
        projectId: string,
        experimentId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<PlanListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/plans',
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
     * Get Plan
     * Return one generated plan's draft + structured experiment report.
     * @param projectId
     * @param experimentId
     * @param runId
     * @param molexpSession
     * @returns PlanDetailResponse Successful Response
     * @throws ApiError
     */
    public static getPlanApiProjectsProjectIdExperimentsExperimentIdPlansRunIdGet(
        projectId: string,
        experimentId: string,
        runId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<PlanDetailResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/plans/{run_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'run_id': runId,
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
