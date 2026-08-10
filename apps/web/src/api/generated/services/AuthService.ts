/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AuthStatusResponse } from '../models/AuthStatusResponse';
import type { AuthTokenResponse } from '../models/AuthTokenResponse';
import type { AuthUserListResponse } from '../models/AuthUserListResponse';
import type { AuthUserPublic } from '../models/AuthUserPublic';
import type { CreateUserRequest } from '../models/CreateUserRequest';
import type { LoginRequest } from '../models/LoginRequest';
import type { PasswordRequest } from '../models/PasswordRequest';
import type { PatchUserRequest } from '../models/PatchUserRequest';
import type { SwitchRequest } from '../models/SwitchRequest';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AuthService {
    /**
     * Auth Login
     * @param requestBody
     * @returns AuthUserPublic Successful Response
     * @throws ApiError
     */
    public static authLoginApiAuthLoginPost(
        requestBody: LoginRequest,
    ): CancelablePromise<AuthUserPublic> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/auth/login',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Auth Logout
     * @param molexpSession
     * @returns void
     * @throws ApiError
     */
    public static authLogoutApiAuthLogoutPost(
        molexpSession?: (string | null),
    ): CancelablePromise<void> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/auth/logout',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Auth Me
     * @param molexpSession
     * @returns AuthUserPublic Successful Response
     * @throws ApiError
     */
    public static authMeApiAuthMeGet(
        molexpSession?: (string | null),
    ): CancelablePromise<AuthUserPublic> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/auth/me',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Auth Refresh
     * @param molexpSession
     * @returns AuthUserPublic Successful Response
     * @throws ApiError
     */
    public static authRefreshApiAuthRefreshPost(
        molexpSession?: (string | null),
    ): CancelablePromise<AuthUserPublic> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/auth/refresh',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Auth Status
     * @param molexpSession
     * @returns AuthStatusResponse Successful Response
     * @throws ApiError
     */
    public static authStatusApiAuthStatusGet(
        molexpSession?: (string | null),
    ): CancelablePromise<AuthStatusResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/auth/status',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Auth Switch
     * @param requestBody
     * @param molexpSession
     * @returns AuthUserPublic Successful Response
     * @throws ApiError
     */
    public static authSwitchApiAuthSwitchPost(
        requestBody: SwitchRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<AuthUserPublic> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/auth/switch',
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
     * Auth Token
     * @param molexpSession
     * @returns AuthTokenResponse Successful Response
     * @throws ApiError
     */
    public static authTokenApiAuthTokenGet(
        molexpSession?: (string | null),
    ): CancelablePromise<AuthTokenResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/auth/token',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Users
     * @param molexpSession
     * @returns AuthUserListResponse Successful Response
     * @throws ApiError
     */
    public static listUsersApiAuthUsersGet(
        molexpSession?: (string | null),
    ): CancelablePromise<AuthUserListResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/auth/users',
            cookies: {
                'molexp_session': molexpSession,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create User
     * @param requestBody
     * @param molexpSession
     * @returns AuthUserPublic Successful Response
     * @throws ApiError
     */
    public static createUserApiAuthUsersPost(
        requestBody: CreateUserRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<AuthUserPublic> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/auth/users',
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
     * Delete User
     * @param username
     * @param molexpSession
     * @returns void
     * @throws ApiError
     */
    public static deleteUserApiAuthUsersUsernameDelete(
        username: string,
        molexpSession?: (string | null),
    ): CancelablePromise<void> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/auth/users/{username}',
            path: {
                'username': username,
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
     * Patch User
     * @param username
     * @param requestBody
     * @param molexpSession
     * @returns AuthUserPublic Successful Response
     * @throws ApiError
     */
    public static patchUserApiAuthUsersUsernamePatch(
        username: string,
        requestBody: PatchUserRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<AuthUserPublic> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/auth/users/{username}',
            path: {
                'username': username,
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
     * Set User Password
     * @param username
     * @param requestBody
     * @param molexpSession
     * @returns AuthUserPublic Successful Response
     * @throws ApiError
     */
    public static setUserPasswordApiAuthUsersUsernamePasswordPost(
        username: string,
        requestBody: PasswordRequest,
        molexpSession?: (string | null),
    ): CancelablePromise<AuthUserPublic> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/auth/users/{username}/password',
            path: {
                'username': username,
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
}
