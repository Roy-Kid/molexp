/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Body for ``POST /api/workspace/cache/{invalidate,refresh}``.
 */
export type CacheControlRequest = {
    /**
     * Drop this entry only (and its descendants if a directory).
     */
    path?: (string | null);
    /**
     * When ``path`` is null: 'all' drops everything; 'indices' drops navigation-index entries only.
     */
    scope?: string;
};

