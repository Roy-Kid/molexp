/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AgentService {
    /**
     * Agent Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathPatch(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/agent/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathPatch1(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathPatch2(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/agent/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathPatch3(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathPatch4(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/agent/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
