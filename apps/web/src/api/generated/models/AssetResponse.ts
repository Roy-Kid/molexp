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
    content_hash?: (string | null);
    created_at: string;
    extra?: Record<string, any>;
    has_preview_sidecar?: boolean;
    id: string;
    kind: string;
    name: string;
    path: string;
    producer?: (Record<string, any> | null);
    scope_ids: Array<string>;
    scope_kind: string;
    tags?: Record<string, string>;
    updated_at: string;
};

