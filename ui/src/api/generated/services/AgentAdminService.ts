/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AgentAdminService {
    /**
     * Agent Admin Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentAdminDisabledApiApiAgentAdminPathDelete(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/api/agent/admin/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Admin Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentAdminDisabledApiApiAgentAdminPathDelete1(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/api/agent/admin/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Admin Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentAdminDisabledApiApiAgentAdminPathDelete2(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/api/agent/admin/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Admin Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentAdminDisabledApiApiAgentAdminPathDelete3(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/api/agent/admin/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Agent Admin Disabled
     * @param path
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentAdminDisabledApiApiAgentAdminPathDelete4(
        path: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/api/agent/admin/{path}',
            path: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
