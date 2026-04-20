/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { UiPluginListResponse } from '../models/UiPluginListResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class PluginsService {
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
