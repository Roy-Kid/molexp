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
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathDelete(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/agent/{path}',
            path: {
                'path': path,
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
     * Agent Disabled
     * @param path
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathDelete1(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/agent/{path}',
            path: {
                'path': path,
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
     * Agent Disabled
     * @param path
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathDelete2(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/agent/{path}',
            path: {
                'path': path,
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
     * Agent Disabled
     * @param path
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathDelete3(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/agent/{path}',
            path: {
                'path': path,
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
     * Agent Disabled
     * @param path
     * @param molexpSession
     * @returns any Successful Response
     * @throws ApiError
     */
    public static agentDisabledApiAgentPathDelete4(
        path: string,
        molexpSession?: (string | null),
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'PUT',
            url: '/api/agent/{path}',
            path: {
                'path': path,
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
