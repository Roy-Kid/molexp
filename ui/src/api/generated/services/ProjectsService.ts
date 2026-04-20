/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AssetResponse } from '../models/AssetResponse';
import type { Body_upload_project_asset_api_projects__id__assets_upload_post } from '../models/Body_upload_project_asset_api_projects__id__assets_upload_post';
import type { MessageResponse } from '../models/MessageResponse';
import type { ProjectCreateRequest } from '../models/ProjectCreateRequest';
import type { ProjectResponse } from '../models/ProjectResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ProjectsService {
    /**
     * List Projects
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static listProjectsApiProjectsGet(): CancelablePromise<Array<ProjectResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects',
        });
    }
    /**
     * Create Project
     * @param requestBody
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static createProjectApiProjectsPost(
        requestBody: ProjectCreateRequest,
    ): CancelablePromise<ProjectResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects',
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Project
     * @param id
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static getProjectApiProjectsIdGet(
        id: string,
    ): CancelablePromise<ProjectResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{id}',
            path: {
                'id': id,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Delete Project
     * @param id
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteProjectApiProjectsIdDelete(
        id: string,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/projects/{id}',
            path: {
                'id': id,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Project Assets
     * List every asset (any kind) in the project scope via the catalog.
     * @param id
     * @param limit
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static listProjectAssetsApiProjectsIdAssetsGet(
        id: string,
        limit: number = 100,
    ): CancelablePromise<Array<AssetResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{id}/assets',
            path: {
                'id': id,
            },
            query: {
                'limit': limit,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Project Asset
     * @param id
     * @param assetId
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static getProjectAssetApiProjectsIdAssetsAssetIdGet(
        id: string,
        assetId: string,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{id}/assets/{asset_id}',
            path: {
                'id': id,
                'asset_id': assetId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Upload Project Asset
     * Upload a file into the project's ``DataAssetLibrary``.
     * @param id
     * @param formData
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static uploadProjectAssetApiProjectsIdAssetsUploadPost(
        id: string,
        formData: Body_upload_project_asset_api_projects__id__assets_upload_post,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{id}/assets/upload',
            path: {
                'id': id,
            },
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Download Project Asset
     * @param id
     * @param assetId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static downloadProjectAssetApiProjectsIdAssetsAssetIdDownloadGet(
        id: string,
        assetId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{id}/assets/{asset_id}/download',
            path: {
                'id': id,
                'asset_id': assetId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
