/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Live remote-index progress for the MolVis-style status bar.
 */
export type CacheStatusResponse = {
    /**
     * False when the active workspace is local
     */
    cached: boolean;
    connected?: (boolean | null);
    done?: number;
    indexed?: (boolean | null);
    indexing?: (boolean | null);
    message?: string;
    percent?: (number | null);
    phase?: string;
    ready?: (boolean | null);
    total?: number;
};

