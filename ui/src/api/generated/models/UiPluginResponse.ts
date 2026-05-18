/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Per-bundle entry returned by ``GET /api/plugins``.
 *
 * Carries no UI semantics — those live in each bundle's own
 * ``manifest.json`` (fetched by the browser-side loader). The shape
 * is deliberately minimal: a stable ``id``, plus the two URLs the
 * frontend needs to fetch the manifest and dynamic-import the entry.
 */
export type UiPluginResponse = {
    id: string;
    manifestUrl: string;
    entryUrl: string;
};
