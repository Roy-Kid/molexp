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
    public static agentDisabledApiApiAgentPathDelete(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/api/agent/{path}',
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
    public static agentDisabledApiApiAgentPathDelete1(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/api/agent/{path}',
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
    public static agentDisabledApiApiAgentPathDelete2(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/api/agent/{path}',
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
    public static agentDisabledApiApiAgentPathDelete3(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/api/agent/{path}',
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
    public static agentDisabledApiApiAgentPathDelete4(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/api/agent/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
