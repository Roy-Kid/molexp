/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AssetLineageResponse } from '../models/AssetLineageResponse';
import type { AssetResponse } from '../models/AssetResponse';
import type { Body_import_data_asset_api_assets_data_import_post } from '../models/Body_import_data_asset_api_assets_data_import_post';
import type { Body_import_data_asset_api_workspaces__ws__assets_data_import_post } from '../models/Body_import_data_asset_api_workspaces__ws__assets_data_import_post';
import type { DataAssetRegisterRequest } from '../models/DataAssetRegisterRequest';
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
    /**
     * Register Data Asset
     * Register an existing workspace file in place as a ``DataAsset``.
     *
     * The file stays where it is — only an index entry is created — so a
     * same-stem preview sidecar remains a real sibling of the resolved path.
     * @param requestBody
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static registerDataAssetApiAssetsDataRegisterPost(
        requestBody: DataAssetRegisterRequest,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/assets/data/register',
            body: requestBody,
            mediaType: 'application/json',
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
     * Preview Asset
     * Render a preview of a sidecar-backed dataset asset.
     *
     * Args:
     * asset_id: Catalog id of the dataset asset.
     * format: ``frames`` (extended-XYZ bytes for the JS trajectory viewer)
     * or ``png`` (headless molvis snapshot).
     * limit: Host-owned cap on the number of frames previewed.
     *
     * Returns:
     * A streaming response — ``chemical/x-xyz`` for ``frames``,
     * ``image/png`` for ``png``.
     *
     * Raises:
     * AssetNotFoundError: Unknown asset id (404).
     * PreviewSidecarNotFoundError: No sidecar next to the dataset (404).
     * NoReaderInSidecarError / AmbiguousReaderError / PreviewReaderError:
     * The sidecar is empty / ambiguous / broken (422).
     * @param assetId
     * @param format
     * @param limit
     * @returns any Successful Response
     * @throws ApiError
     */
    public static previewAssetApiAssetsAssetIdPreviewGet(
        assetId: string,
        format: 'frames' | 'png' = 'frames',
        limit: number = 200,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/assets/{asset_id}/preview',
            path: {
                'asset_id': assetId,
            },
            query: {
                'format': format,
                'limit': limit,
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
     * List Assets
     * Query assets from the workspace catalog with optional filters.
     * @param ws
     * @param kind
     * @param scopeKind
     * @param runId
     * @param taskId
     * @param limit
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static listAssetsApiWorkspacesWsAssetsGet(
        ws: string,
        kind?: (string | null),
        scopeKind?: (string | null),
        runId?: (string | null),
        taskId?: (string | null),
        limit: number = 100,
    ): CancelablePromise<Array<AssetResponse>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/assets',
            path: {
                'ws': ws,
            },
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
     * Import Data Asset
     * Upload a file and register it as a workspace-scoped ``DataAsset``.
     * @param ws
     * @param formData
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static importDataAssetApiWorkspacesWsAssetsDataImportPost(
        ws: string,
        formData: Body_import_data_asset_api_workspaces__ws__assets_data_import_post,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/assets/data/import',
            path: {
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
     * Register Data Asset
     * Register an existing workspace file in place as a ``DataAsset``.
     *
     * The file stays where it is — only an index entry is created — so a
     * same-stem preview sidecar remains a real sibling of the resolved path.
     * @param ws
     * @param requestBody
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static registerDataAssetApiWorkspacesWsAssetsDataRegisterPost(
        ws: string,
        requestBody: DataAssetRegisterRequest,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/workspaces/{ws}/assets/data/register',
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
     * Get Asset
     * @param assetId
     * @param ws
     * @returns AssetResponse Successful Response
     * @throws ApiError
     */
    public static getAssetApiWorkspacesWsAssetsAssetIdGet(
        assetId: string,
        ws: string,
    ): CancelablePromise<AssetResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/assets/{asset_id}',
            path: {
                'asset_id': assetId,
                'ws': ws,
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
     * @param ws
     * @returns any Successful Response
     * @throws ApiError
     */
    public static assetContentApiWorkspacesWsAssetsAssetIdContentGet(
        assetId: string,
        ws: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/assets/{asset_id}/content',
            path: {
                'asset_id': assetId,
                'ws': ws,
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
     * @param ws
     * @returns AssetLineageResponse Successful Response
     * @throws ApiError
     */
    public static getAssetLineageApiWorkspacesWsAssetsAssetIdLineageGet(
        assetId: string,
        ws: string,
    ): CancelablePromise<AssetLineageResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/assets/{asset_id}/lineage',
            path: {
                'asset_id': assetId,
                'ws': ws,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Preview Asset
     * Render a preview of a sidecar-backed dataset asset.
     *
     * Args:
     * asset_id: Catalog id of the dataset asset.
     * format: ``frames`` (extended-XYZ bytes for the JS trajectory viewer)
     * or ``png`` (headless molvis snapshot).
     * limit: Host-owned cap on the number of frames previewed.
     *
     * Returns:
     * A streaming response — ``chemical/x-xyz`` for ``frames``,
     * ``image/png`` for ``png``.
     *
     * Raises:
     * AssetNotFoundError: Unknown asset id (404).
     * PreviewSidecarNotFoundError: No sidecar next to the dataset (404).
     * NoReaderInSidecarError / AmbiguousReaderError / PreviewReaderError:
     * The sidecar is empty / ambiguous / broken (422).
     * @param assetId
     * @param ws
     * @param format
     * @param limit
     * @returns any Successful Response
     * @throws ApiError
     */
    public static previewAssetApiWorkspacesWsAssetsAssetIdPreviewGet(
        assetId: string,
        ws: string,
        format: 'frames' | 'png' = 'frames',
        limit: number = 200,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/assets/{asset_id}/preview',
            path: {
                'asset_id': assetId,
                'ws': ws,
            },
            query: {
                'format': format,
                'limit': limit,
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
     * @param ws
     * @param n
     * @returns string Successful Response
     * @throws ApiError
     */
    public static assetTailApiWorkspacesWsAssetsAssetIdTailGet(
        assetId: string,
        ws: string,
        n: number = 100,
    ): CancelablePromise<string> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/workspaces/{ws}/assets/{asset_id}/tail',
            path: {
                'asset_id': assetId,
                'ws': ws,
            },
            query: {
                'n': n,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
