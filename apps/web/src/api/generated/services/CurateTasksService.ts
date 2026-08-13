/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CurateTaskCreateRequest } from '../models/CurateTaskCreateRequest';
import type { CurateTaskListResponse } from '../models/CurateTaskListResponse';
import type { CurateTaskResponse } from '../models/CurateTaskResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class CurateTasksService {
    /**
     * List Curate Tasks
     * List the live curate tasks in this workspace (in-memory; MVP).
     * @param projectId
     * @param experimentId
     * @param molexpSession
     * @returns CurateTaskListResponse Successful Response
     * @throws ApiError
     */
    public static listCurateTasksApiProjectsProjectIdExperimentsExperimentIdCurateTasksGet(
        projectId: string,
        experimentId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<CurateTaskListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/curate-tasks',
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
     * Create Curate Task
     * Start the curation flow on a content-addressed run under the experiment.
     *
     * Async so the spawned background ``asyncio.Task`` (the curation flow) attaches
     * to the app event loop; the handler itself does no awaiting and returns the
     * initial ``running`` status immediately.
     * @param projectId
     * @param experimentId
     * @param requestBody
     * @param molexpSession
     * @returns CurateTaskResponse Successful Response
     * @throws ApiError
     */
    public static createCurateTaskApiProjectsProjectIdExperimentsExperimentIdCurateTasksPost(
        projectId: string,
        experimentId: string,
        requestBody: CurateTaskCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<CurateTaskResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/curate-tasks',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
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
     * Get Curate Task
     * Return one curate task's current status.
     * @param projectId
     * @param experimentId
     * @param taskId
     * @param molexpSession
     * @returns CurateTaskResponse Successful Response
     * @throws ApiError
     */
    public static getCurateTaskApiProjectsProjectIdExperimentsExperimentIdCurateTasksTaskIdGet(
        projectId: string,
        experimentId: string,
        taskId: string,
        molexpSession?: (string | null),
    ): CancelablePromise<CurateTaskResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/experiments/{experiment_id}/curate-tasks/{task_id}',
            path: {
                'project_id': projectId,
                'experiment_id': experimentId,
                'task_id': taskId,
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
