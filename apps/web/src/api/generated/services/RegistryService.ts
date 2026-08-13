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
     * List Plugins
     * List entry-point–discovered UI bundles.
     *
     * Built-in plugins (``core``, ``molplot``, ``molq``, ``molvis``, …) are
     * statically imported by the frontend and do **not** appear here. There
     * is no metrics product plugin — plots are molplot only. The response
     * carries no UI semantics — those live in each bundle's own
     * ``manifest.json``, fetched by the browser-side loader.
     * @param molexpSession
     * @returns UiPluginListResponse Successful Response
     * @throws ApiError
     */
    public static listPluginsApiPluginsGet(
        molexpSession?: (string | null),
    ): CancelablePromise<UiPluginListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/plugins',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Task Types
     * Return every task-type slug the agent / UI can compose into IR.
     * @param molexpSession
     * @returns TaskTypeListResponse Successful Response
     * @throws ApiError
     */
    public static listTaskTypesApiTasksGet(
        molexpSession?: (string | null),
    ): CancelablePromise<TaskTypeListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/tasks',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Task Type
     * Return one task type by slug, or 404 if not registered.
     * @param slug
     * @param molexpSession
     * @returns TaskTypeResponse Successful Response
     * @throws ApiError
     */
    public static getTaskTypeApiTasksSlugGet(
        slug: string,
        molexpSession?: (string | null),
    ): CancelablePromise<TaskTypeResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/tasks/{slug}',
            path: {
                'slug': slug,
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
