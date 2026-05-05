/**
 * Mock handlers for the unified /api/assets surface.
 *
 * The real backend is catalog-driven and returns a discriminated union of
 * typed Asset kinds (data / artifact / log / checkpoint / …). These handlers
 * emulate the query filters and per-kind endpoints (`content`, `tail`,
 * `data/import`).
 */

import { http, HttpResponse } from "msw";
import { getAllAssets, getAsset, setAsset } from "../db";
import type { ApiAssetResponse } from "../../src/app/types";

interface LineageNodeShape {
    id: string;
    name: string;
    kind: string;
    scope_kind: string;
}

const _node = (a: ApiAssetResponse): LineageNodeShape => ({
    id: a.id,
    name: a.name,
    kind: a.kind,
    scope_kind: a.scope_kind,
});

const _ancestorIds = (assetId: string): Set<string> => {
    const visited = new Set<string>();
    const frontier = [assetId];
    while (frontier.length > 0) {
        const cur = frontier.pop()!;
        const a = getAsset(cur);
        if (!a || !a.producer) continue;
        const inputs = (a.producer as Record<string, unknown>).inputs as
            | string[]
            | undefined;
        for (const upstream of inputs ?? []) {
            if (upstream === assetId || visited.has(upstream)) continue;
            visited.add(upstream);
            frontier.push(upstream);
        }
    }
    return visited;
};

const _descendantIds = (assetId: string): Set<string> => {
    const all = getAllAssets();
    const childrenOf = new Map<string, string[]>();
    for (const a of all) {
        const inputs = (a.producer as Record<string, unknown> | undefined)
            ?.inputs as string[] | undefined;
        for (const inp of inputs ?? []) {
            const arr = childrenOf.get(inp) ?? [];
            arr.push(a.id);
            childrenOf.set(inp, arr);
        }
    }
    const visited = new Set<string>();
    const frontier = [assetId];
    while (frontier.length > 0) {
        const cur = frontier.pop()!;
        for (const child of childrenOf.get(cur) ?? []) {
            if (child === assetId || visited.has(child)) continue;
            visited.add(child);
            frontier.push(child);
        }
    }
    return visited;
};

const API_BASE = "/api";

const applyFilters = (
    assets: ApiAssetResponse[],
    params: URLSearchParams,
): ApiAssetResponse[] => {
    const kind = params.get("kind");
    const scopeKind = params.get("scope_kind");
    const runId = params.get("run_id");
    const taskId = params.get("task_id");
    const limitRaw = params.get("limit");
    const limit = limitRaw ? Number.parseInt(limitRaw, 10) : 100;

    let filtered = assets;
    if (kind) filtered = filtered.filter((a) => a.kind === kind);
    if (scopeKind) filtered = filtered.filter((a) => a.scope_kind === scopeKind);
    if (runId) filtered = filtered.filter((a) => a.producer?.run_id === runId);
    if (taskId) filtered = filtered.filter((a) => a.producer?.task_id === taskId);
    return filtered.slice(0, Number.isFinite(limit) ? limit : 100);
};

export const assetHandlers = [
    // GET /api/assets - query the catalog with filters
    http.get(`${API_BASE}/assets`, ({ request }) => {
        const url = new URL(request.url);
        return HttpResponse.json(applyFilters(getAllAssets(), url.searchParams));
    }),

    // GET /api/assets/:assetId - asset detail (discriminated by kind)
    http.get(`${API_BASE}/assets/:assetId`, ({ params }) => {
        const asset = getAsset(params.assetId as string);
        if (!asset) {
            return HttpResponse.json({ detail: "Asset not found" }, { status: 404 });
        }
        return HttpResponse.json(asset);
    }),

    // GET /api/assets/:assetId/lineage - upstream + downstream neighbors
    http.get(`${API_BASE}/assets/:assetId/lineage`, ({ params }) => {
        const assetId = params.assetId as string;
        const target = getAsset(assetId);
        if (!target) {
            return HttpResponse.json({ detail: "Asset not found" }, { status: 404 });
        }
        const ancestors = [..._ancestorIds(assetId)]
            .sort()
            .map((id) => getAsset(id))
            .filter((a): a is ApiAssetResponse => Boolean(a))
            .map(_node);
        const descendants = [..._descendantIds(assetId)]
            .sort()
            .map((id) => getAsset(id))
            .filter((a): a is ApiAssetResponse => Boolean(a))
            .map(_node);
        return HttpResponse.json({ asset_id: assetId, ancestors, descendants });
    }),

    // GET /api/assets/:assetId/content - download raw bytes
    http.get(`${API_BASE}/assets/:assetId/content`, ({ params }) => {
        const asset = getAsset(params.assetId as string);
        if (!asset) {
            return HttpResponse.json({ detail: "Asset not found" }, { status: 404 });
        }
        const mime =
            (asset.extra as Record<string, unknown> | undefined)?.mime as
                | string
                | undefined;
        const filename = asset.path.split("/").pop() ?? asset.name;
        return new HttpResponse(new Blob([]), {
            headers: {
                "Content-Type": mime ?? "application/octet-stream",
                "Content-Disposition": `attachment; filename="${filename}"`,
            },
        });
    }),

    // GET /api/assets/:assetId/tail - last N lines (LogAsset only)
    http.get(`${API_BASE}/assets/:assetId/tail`, ({ params, request }) => {
        const asset = getAsset(params.assetId as string);
        if (!asset) {
            return HttpResponse.json({ detail: "Asset not found" }, { status: 404 });
        }
        if (asset.kind !== "log") {
            return HttpResponse.json(
                { detail: `tail only supported for log assets (got ${asset.kind})` },
                { status: 400 },
            );
        }
        const url = new URL(request.url);
        const n = Number.parseInt(url.searchParams.get("n") ?? "100", 10) || 100;
        const placeholder = Array.from({ length: Math.min(n, 5) }, (_, i) =>
            `[mock] log line ${i + 1} for ${asset.name}`
        );
        return new HttpResponse(placeholder.join("\n"), {
            headers: { "Content-Type": "text/plain" },
        });
    }),

    // POST /api/assets/data/import - upload and register a workspace DataAsset
    http.post(`${API_BASE}/assets/data/import`, async ({ request }) => {
        const form = await request.formData();
        const file = form.get("file") as File | null;
        const filename = file?.name ?? "uploaded";
        const size = file?.size ?? 0;
        const now = new Date().toISOString();
        const assetId = `asset-${Date.now()}`;
        const asset: ApiAssetResponse = {
            id: assetId,
            name: filename,
            kind: "data",
            scope_kind: "workspace",
            scope_ids: [],
            path: `data_assets/${assetId}/payload`,
            created_at: now,
            updated_at: now,
            producer: null,
            tags: { original_filename: filename },
            extra: {
                mime: file?.type || "application/octet-stream",
                size,
                source_path: filename,
                import_action: "move",
            },
        };
        setAsset(asset);
        return HttpResponse.json(asset, { status: 201 });
    }),
];
