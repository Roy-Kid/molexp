/**
 * Mock handlers for Asset API endpoints
 */

import { http, HttpResponse } from "msw";
import { getAllAssets } from "../db";

const API_BASE = "/api";

export const assetHandlers = [
    // GET /api/assets - List all assets
    http.get(`${API_BASE}/assets`, () => {
        const assets = getAllAssets();
        return HttpResponse.json(assets);
    }),
];
