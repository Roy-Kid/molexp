/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { UiPluginListResponse } from '../models/UiPluginListResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class RegistryService {
    /**
     * List Nodes
     * @returns any Successful Response
     * @throws ApiError
     */
    public static listNodesApiTasksGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/tasks',
        });
    }
    /**
     * Get Node
     * @param nodeId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getNodeApiTasksNodeIdGet(
        nodeId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/tasks/{node_id}',
            path: {
                'node_id': nodeId,
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
