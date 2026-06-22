/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { PlanTaskCreateRequest } from '../models/PlanTaskCreateRequest';
import type { PlanTaskListResponse } from '../models/PlanTaskListResponse';
import type { PlanTaskResponse } from '../models/PlanTaskResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class PlanTasksService {
    /**
     * List Plan Tasks
     * List the live plan tasks in this workspace (in-memory; MVP).
     * @param projectId
     * @param experimentId
     * @returns PlanTaskListResponse Successful Response
     * @throws ApiError
     */
    public static listPlanTasksApiProjectsProjectIdExperimentsExperimentIdPlanTasksGet(
        projectId: string,
        experimentId: string,
    ): CancelablePromise<PlanTaskListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/plan-tasks',
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
     * Create Plan Task
     * Start a PlanMode pipeline on a content-addressed run under the experiment.
     *
     * Async so the spawned background ``asyncio.Task`` (the PlanMode run) attaches
     * to the app event loop; the handler itself does no awaiting and returns the
     * initial ``running`` status immediately.
     * @param projectId
     * @param experimentId
     * @param requestBody
     * @returns PlanTaskResponse Successful Response
     * @throws ApiError
     */
    public static createPlanTaskApiProjectsProjectIdExperimentsExperimentIdPlanTasksPost(
        projectId: string,
        experimentId: string,
        requestBody: PlanTaskCreateRequest,
    ): CancelablePromise<PlanTaskResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/plan-tasks',
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
     * Get Plan Task
     * Return one plan task's current status.
     * @param projectId
     * @param experimentId
     * @param taskId
     * @returns PlanTaskResponse Successful Response
     * @throws ApiError
     */
    public static getPlanTaskApiProjectsProjectIdExperimentsExperimentIdPlanTasksTaskIdGet(
        projectId: string,
        experimentId: string,
        taskId: string,
    ): CancelablePromise<PlanTaskResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/plan-tasks/{task_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'task_id': taskId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
