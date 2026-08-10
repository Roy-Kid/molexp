/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { TargetCreateRequest } from '../models/TargetCreateRequest';
import type { TargetListResponse } from '../models/TargetListResponse';
import type { TargetResponse } from '../models/TargetResponse';
import type { TargetTestResponse } from '../models/TargetTestResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class TargetsService {
    /**
     * List Targets Endpoint
     * List compute targets — the registered ones plus the built-in ``local``.
     * @param molexpSession
     * @returns TargetListResponse Successful Response
     * @throws ApiError
     */
    public static listTargetsEndpointApiTargetsGet(
        molexpSession?: (string | null),
    ): CancelablePromise<TargetListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/targets',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Target Endpoint
     * Register a new compute target.
     *
     * Mirrors ``molexp target add NAME --scratch ... [--host ...] [--scheduler ...]``.
     * @param requestBody
     * @param molexpSession
     * @returns TargetResponse Successful Response
     * @throws ApiError
     */
    public static createTargetEndpointApiTargetsPost(
        requestBody: TargetCreateRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<TargetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/targets',
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
     * Delete Target Endpoint
     * Remove the named compute target from the workspace registry.
     * @param name
     * @param molexpSession
     * @returns void
     * @throws ApiError
     */
    public static deleteTargetEndpointApiTargetsNameDelete(
        name: string,
        molexpSession?: (string | null),
    ): CancelablePromise<void> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/targets/{name}',
            path: {
                'name': name,
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
     * Test Target Endpoint
     * Verify connectivity to a target — runs the same round-trip probe as
     * ``molexp target test`` (true / mkdir scratch / 1-byte file round-trip).
     *
     * Returns ``ok=False`` with the failing step's detail rather than raising,
     * so the UI can render the failure inline.
     * @param name
     * @param molexpSession
     * @returns TargetTestResponse Successful Response
     * @throws ApiError
     */
    public static testTargetEndpointApiTargetsNameTestPost(
        name: string,
        molexpSession?: (string | null),
    ): CancelablePromise<TargetTestResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/targets/{name}/test',
            path: {
                'name': name,
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
