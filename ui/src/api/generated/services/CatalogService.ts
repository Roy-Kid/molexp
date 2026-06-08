/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CatalogByPathResponse } from '../models/CatalogByPathResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class CatalogService {
    /**
     * Catalog By Path
     * Reverse lookup: find the producer for a given workspace path.
     *
     * Accepts either an absolute path (must be inside the workspace) or a
     * workspace-relative path. Rejects absolute paths outside the workspace
     * root with HTTP 400.
     * @param path Workspace-relative or absolute path
     * @returns CatalogByPathResponse Successful Response
     * @throws ApiError
     */
    public static catalogByPathApiCatalogByPathGet(
        path: string,
    ): CancelablePromise<CatalogByPathResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/catalog/by-path',
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Catalog By Path
     * Reverse lookup: find the producer for a given workspace path.
     *
     * Accepts either an absolute path (must be inside the workspace) or a
     * workspace-relative path. Rejects absolute paths outside the workspace
     * root with HTTP 400.
     * @param ws
     * @param path Workspace-relative or absolute path
     * @returns CatalogByPathResponse Successful Response
     * @throws ApiError
     */
    public static catalogByPathApiWorkspacesWsCatalogByPathGet(
        ws: string,
        path: string,
    ): CancelablePromise<CatalogByPathResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/catalog/by-path',
            path: {
                'ws': ws,
            },
            query: {
                'path': path,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
