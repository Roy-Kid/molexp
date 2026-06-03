/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CatalogProducerInfo } from './CatalogProducerInfo';
import type { CatalogScopeInfo } from './CatalogScopeInfo';
import type { CatalogSibling } from './CatalogSibling';
/**
 * Reverse-lookup: which run/experiment/project produced a file?
 */
export type CatalogByPathResponse = {
    assetId?: (string | null);
    assetKind?: (string | null);
    matched: boolean;
    producer?: (CatalogProducerInfo | null);
    scope?: (CatalogScopeInfo | null);
    siblings?: Array<CatalogSibling>;
    workspaceRelPath: string;
};

