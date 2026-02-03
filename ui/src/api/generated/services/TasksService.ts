/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class TasksService {
    /**
     * List Nodes
     * List all available node types from plugins.
     *
     * Returns:
     * Dictionary with all node definitions including metadata and config schemas
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
     * Get details for a specific node type.
     *
     * Args:
     * node_id: Task identifier (e.g., "io.write_file")
     *
     * Returns:
     * Task definition with metadata and config schema
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
}
