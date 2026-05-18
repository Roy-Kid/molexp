/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * One node in an asset's lineage neighborhood.
 *
 * Carries just enough to render a clickable card in the UI; full
 * asset detail is available via ``GET /api/assets/{id}``.
 */
export type AssetLineageNode = {
    id: string;
    name: string;
    kind: string;
    scope_kind: string;
};
