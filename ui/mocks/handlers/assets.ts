/**
 * Mock handlers for Asset API endpoints
 */

import { http, HttpResponse } from "msw";
import { getAllAssets, getAsset } from "../db";

const API_BASE = "/api";

export const assetHandlers = [
    // GET /api/assets - List all assets
    http.get(`${API_BASE}/assets`, () => {
        const assets = getAllAssets();
        return HttpResponse.json(assets);
    }),

    // GET /api/assets/:assetId - Asset detail
    http.get(`${API_BASE}/assets/:assetId`, ({ params }) => {
        const asset = getAsset(params.assetId as string);
        if (!asset) {
            return HttpResponse.json({ detail: "Asset not found" }, { status: 404 });
        }
        return HttpResponse.json(asset);
    }),

    // GET /api/assets/:assetId/download - Binary asset download placeholder
    http.get(`${API_BASE}/assets/:assetId/download`, ({ params }) => {
        const asset = getAsset(params.assetId as string);
        if (!asset) {
            return HttpResponse.json({ detail: "Asset not found" }, { status: 404 });
        }
        return new HttpResponse(new Blob([]), {
            headers: {
                "Content-Type": asset.mimeType ?? "application/octet-stream",
                "Content-Disposition": `attachment; filename=\"${asset.assetId}.${asset.format}\"`,
            },
        });
    }),
];
