/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TaskTypeListResponse } from '../models/TaskTypeListResponse';
import type { TaskTypeResponse } from '../models/TaskTypeResponse';
import type { UiPluginListResponse } from '../models/UiPluginListResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class RegistryService {
    /**
     * List Task Types
     * Return every task-type slug the agent / UI can compose into IR.
     * @returns TaskTypeListResponse Successful Response
     * @throws ApiError
     */
    public static listTaskTypesApiTasksGet(): CancelablePromise<TaskTypeListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/tasks',
        });
    }
    /**
     * Get Task Type
     * Return one task type by slug, or 404 if not registered.
     * @param slug
     * @returns TaskTypeResponse Successful Response
     * @throws ApiError
     */
    public static getTaskTypeApiTasksSlugGet(
        slug: string,
    ): CancelablePromise<TaskTypeResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/tasks/{slug}',
            path: {
                'slug': slug,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Plugins
     * @returns UiPluginListResponse Successful Response
     * @throws ApiError
     */
    public static listPluginsApiPluginsGet(): CancelablePromise<UiPluginListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins',
        });
    }
}
