/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { AssetFileResponse } from './AssetFileResponse';
/**
 * Asset response model.
 */
export type AssetResponse = {
    /**
     * ISO 8601 creation timestamp
     */
    created: string;
    id: string;
    assetId: string;
    type: string;
    format: string;
    size: number;
    contentHash: string;
    mimeType?: string;
    producerRunId?: (string | null);
    tags?: Array<string>;
    metadata?: Record<string, any>;
    files?: Array<AssetFileResponse>;
};

