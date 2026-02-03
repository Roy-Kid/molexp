/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AssetResponse } from '../models/AssetResponse';
import type { Body_upload_asset_api_assets_upload_post } from '../models/Body_upload_asset_api_assets_upload_post';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AssetsService {
    /**
     * List Assets
     * List all assets.
     * @param limit
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static listAssetsApiAssetsGet(
        limit: number = 100,
    ): CancelablePromise<Array<AssetResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/assets',
            query: {
                'limit': limit,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Asset
     * Get asset details.
     * @param assetId
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static getAssetApiAssetsAssetIdGet(
        assetId: string,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/assets/{asset_id}',
            path: {
                'asset_id': assetId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Upload Asset
     * Upload a new asset.
     * @param formData
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static uploadAssetApiAssetsUploadPost(
        formData: Body_upload_asset_api_assets_upload_post,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/assets/upload',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Download Asset
     * Download asset content.
     * @param assetId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static downloadAssetApiAssetsAssetIdDownloadGet(
        assetId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/assets/{asset_id}/download',
            path: {
                'asset_id': assetId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
