/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * PUT body — only submitted fields are written.
 *
 * ``models`` is the **global** tier table: each value should be a full
 * ``provider:model`` id (or a bare id + ``provider`` for same-provider legacy).
 * Credential fields still use ``provider`` + ``api_key`` / ``base_url``.
 */
export type ProviderUpdateRequest = {
    api_key?: (string | null);
    base_url?: (string | null);
    instructions?: (string | null);
    model?: (string | null);
    models?: (Record<string, string> | null);
    provider?: (string | null);
};

