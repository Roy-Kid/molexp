/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AssetLineageResponse } from '../models/AssetLineageResponse';
import type { AssetResponse } from '../models/AssetResponse';
import type { Body_import_data_asset_api_assets_data_import_post } from '../models/Body_import_data_asset_api_assets_data_import_post';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class AssetsService {
    /**
     * List Assets
     * Query assets from the workspace catalog with optional filters.
     * @param kind
     * @param scopeKind
     * @param runId
     * @param taskId
     * @param limit
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static listAssetsApiAssetsGet(
        kind?: (string | null),
        scopeKind?: (string | null),
        runId?: (string | null),
        taskId?: (string | null),
        limit: number = 100,
    ): CancelablePromise<Array<AssetResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/assets',
            query: {
                'kind': kind,
                'scope_kind': scopeKind,
                'run_id': runId,
                'task_id': taskId,
                'limit': limit,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Asset
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
     * Get Asset Lineage
     * Return the asset's transitive ancestors and descendants.
     *
     * Walks the ``Producer.inputs`` DAG built by run-time tasks that
     * declare ``consumed=[...]`` on artifact / data registration. The
     * starting asset is excluded from both lists.
     * @param assetId
     * @returns AssetLineageResponse Successful Response
     * @throws ApiError
     */
    public static getAssetLineageApiAssetsAssetIdLineageGet(
        assetId: string,
    ): CancelablePromise<AssetLineageResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/assets/{asset_id}/lineage',
            path: {
                'asset_id': assetId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Asset Content
     * Download the asset's file content.
     * @param assetId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static assetContentApiAssetsAssetIdContentGet(
        assetId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/assets/{asset_id}/content',
            path: {
                'asset_id': assetId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Asset Tail
     * Return the last N lines (``LogAsset`` only).
     * @param assetId
     * @param n
     * @returns string Successful Response
     * @throws ApiError
     */
    public static assetTailApiAssetsAssetIdTailGet(
        assetId: string,
        n: number = 100,
    ): CancelablePromise<string> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/assets/{asset_id}/tail',
            path: {
                'asset_id': assetId,
            },
            query: {
                'n': n,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Import Data Asset
     * Upload a file and register it as a workspace-scoped ``DataAsset``.
     * @param formData
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static importDataAssetApiAssetsDataImportPost(
        formData: Body_import_data_asset_api_assets_data_import_post,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/assets/data/import',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
