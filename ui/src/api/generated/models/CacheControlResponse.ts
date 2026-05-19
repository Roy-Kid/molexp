/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type CacheControlResponse = {
    /**
     * Number of cache entries removed
     */
    dropped: number;
    /**
     * Per-node warnings raised by the post-invalidate refresh (refresh endpoint only).
     */
    warnings?: Array<string>;
};
