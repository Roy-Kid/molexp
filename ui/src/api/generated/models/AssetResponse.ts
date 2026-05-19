/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Serialized typed ``Asset``.
 *
 * ``kind`` is the discriminator (``data`` / ``artifact`` / ``log`` / …).
 * ``extra`` carries subclass-specific fields so the frontend can render
 * per-kind details without a separate schema per kind.
 * ``content_hash`` is the sha256 (``"sha256:<hex>"``) of the payload
 * when the asset is content-addressable; ``None`` for streaming kinds.
 */
export type AssetResponse = {
    id: string;
    name: string;
    kind: string;
    scope_kind: string;
    scope_ids: Array<string>;
    path: string;
    created_at: string;
    updated_at: string;
    producer?: (Record<string, any> | null);
    tags?: Record<string, string>;
    extra?: Record<string, any>;
    content_hash?: (string | null);
};

