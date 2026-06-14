/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AssetResponse } from '../models/AssetResponse';
import type { Body_upload_project_asset_api_projects__project_id__assets_upload_post } from '../models/Body_upload_project_asset_api_projects__project_id__assets_upload_post';
import type { Body_upload_project_asset_api_workspaces__ws__projects__project_id__assets_upload_post } from '../models/Body_upload_project_asset_api_workspaces__ws__projects__project_id__assets_upload_post';
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
     * Delete Project
     * @param projectId
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteProjectApiProjectsProjectIdDelete(
        projectId: string,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/projects/{project_id}',
            path: {
                'project_id': projectId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Project
     * @param projectId
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static getProjectApiProjectsProjectIdGet(
        projectId: string,
    ): CancelablePromise<ProjectResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}',
            path: {
                'project_id': projectId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Project Assets
     * List every asset (any kind) in the project scope via the catalog.
     * @param projectId
     * @param limit
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static listProjectAssetsApiProjectsProjectIdAssetsGet(
        projectId: string,
        limit: number = 100,
    ): CancelablePromise<Array<AssetResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/assets',
            path: {
                'project_id': projectId,
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
     * Upload Project Asset
     * Upload a file into the project's ``DataAssetLibrary``.
     * @param projectId
     * @param formData
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static uploadProjectAssetApiProjectsProjectIdAssetsUploadPost(
        projectId: string,
        formData: Body_upload_project_asset_api_projects__project_id__assets_upload_post,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/projects/{project_id}/assets/upload',
            path: {
                'project_id': projectId,
            },
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Project Asset
     * @param projectId
     * @param assetId
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static getProjectAssetApiProjectsProjectIdAssetsAssetIdGet(
        projectId: string,
        assetId: string,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/assets/{asset_id}',
            path: {
                'project_id': projectId,
                'asset_id': assetId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Download Project Asset
     * @param projectId
     * @param assetId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static downloadProjectAssetApiProjectsProjectIdAssetsAssetIdDownloadGet(
        projectId: string,
        assetId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/projects/{project_id}/assets/{asset_id}/download',
            path: {
                'project_id': projectId,
                'asset_id': assetId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Projects
     * @param ws
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static listProjectsApiWorkspacesWsProjectsGet(
        ws: string,
    ): CancelablePromise<Array<ProjectResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects',
            path: {
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Create Project
     * @param ws
     * @param requestBody
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static createProjectApiWorkspacesWsProjectsPost(
        ws: string,
        requestBody: ProjectCreateRequest,
    ): CancelablePromise<ProjectResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects',
            path: {
                'ws': ws,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Delete Project
     * @param projectId
     * @param ws
     * @returns MessageResponse Successful Response
     * @throws ApiError
     */
    public static deleteProjectApiWorkspacesWsProjectsProjectIdDelete(
        projectId: string,
        ws: string,
    ): CancelablePromise<MessageResponse> {
        return __request(OpenAPI, {
            method: 'DELETE',
            url: '/api/workspaces/{ws}/projects/{project_id}',
            path: {
                'project_id': projectId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Project
     * @param projectId
     * @param ws
     * @returns ProjectResponse Successful Response
     * @throws ApiError
     */
    public static getProjectApiWorkspacesWsProjectsProjectIdGet(
        projectId: string,
        ws: string,
    ): CancelablePromise<ProjectResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}',
            path: {
                'project_id': projectId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * List Project Assets
     * List every asset (any kind) in the project scope via the catalog.
     * @param projectId
     * @param ws
     * @param limit
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static listProjectAssetsApiWorkspacesWsProjectsProjectIdAssetsGet(
        projectId: string,
        ws: string,
        limit: number = 100,
    ): CancelablePromise<Array<AssetResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/assets',
            path: {
                'project_id': projectId,
                'ws': ws,
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
     * Upload Project Asset
     * Upload a file into the project's ``DataAssetLibrary``.
     * @param projectId
     * @param ws
     * @param formData
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static uploadProjectAssetApiWorkspacesWsProjectsProjectIdAssetsUploadPost(
        projectId: string,
        ws: string,
        formData: Body_upload_project_asset_api_workspaces__ws__projects__project_id__assets_upload_post,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/projects/{project_id}/assets/upload',
            path: {
                'project_id': projectId,
                'ws': ws,
            },
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Project Asset
     * @param projectId
     * @param assetId
     * @param ws
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static getProjectAssetApiWorkspacesWsProjectsProjectIdAssetsAssetIdGet(
        projectId: string,
        assetId: string,
        ws: string,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/assets/{asset_id}',
            path: {
                'project_id': projectId,
                'asset_id': assetId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Download Project Asset
     * @param projectId
     * @param assetId
     * @param ws
     * @returns any Successful Response
     * @throws ApiError
     */
    public static downloadProjectAssetApiWorkspacesWsProjectsProjectIdAssetsAssetIdDownloadGet(
        projectId: string,
        assetId: string,
        ws: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/projects/{project_id}/assets/{asset_id}/download',
            path: {
                'project_id': projectId,
                'asset_id': assetId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
