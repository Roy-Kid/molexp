/**
 * Mock handlers for Workspace API endpoints
 */

import { http, HttpResponse } from "msw";
import {
    createDirectory,
    deleteFile,
    getAllAssets,
    getAllProjects,
    getFile,
    getFileTree,
    writeFile,
} from "../db";
import type { FileNode } from "../db";

const API_BASE = "/api";

/**
 * Build nested file tree from flat file map
 */
function buildNestedTree(path: string): { path: string; children: FileNode[] } {
    if (path === "/" || path === "") {
        // Root level
        const rootFiles = getFileTree();
        return {
            path: "/",
            children: rootFiles,
        };
    }

    // Find the requested node
    const node = getFile(path);
    if (!node) {
        return { path, children: [] };
    }

    if (node.type === "file") {
        return { path, children: [] };
    }

    return {
        path,
        children: node.children || [],
    };
}

export const workspaceHandlers = [
    // GET /api/workspace/info - Get workspace metadata
    http.get(`${API_BASE}/workspace/info`, () => {
        const projects = getAllProjects();
        const assets = getAllAssets();

        return HttpResponse.json({
            root: "/mock-workspace",
            projectCount: projects.length,
            assetCount: assets.length,
        });
    }),

    // GET /api/workspace/files - List workspace files
    http.get(`${API_BASE}/workspace/files`, ({ request }) => {
        const url = new URL(request.url);
        const path = url.searchParams.get("path") || "/";

        const tree = buildNestedTree(path);
        return HttpResponse.json(tree);
    }),

    // GET /api/workspace/file - Read file content (text)
    http.get(`${API_BASE}/workspace/file`, ({ request }) => {
        const url = new URL(request.url);
        const path = url.searchParams.get("path") || "";

        const file = getFile(path);
        if (!file || file.type !== "file") {
            return HttpResponse.json(
                { message: "File not found" },
                { status: 404 }
            );
        }

        return HttpResponse.json({
            content: file.content || "",
        });
    }),

    // GET /api/files - Compatibility endpoint used by document resolvers
    http.get(`${API_BASE}/files`, ({ request }) => {
        const url = new URL(request.url);
        const path = url.searchParams.get("path") || "";
        const file = getFile(path);

        if (!file || file.type !== "file") {
            return HttpResponse.json({ message: "File not found" }, { status: 404 });
        }

        const content = file.content || "";
        const trimmed = content.trim();

        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            try {
                return HttpResponse.json(JSON.parse(content));
            } catch {
                return HttpResponse.json({ content });
            }
        }

        return HttpResponse.json({ content });
    }),

    // GET /api/workspace/file/blob - Read file content (binary)
    http.get(`${API_BASE}/workspace/file/blob`, ({ request }) => {
        const url = new URL(request.url);
        const path = url.searchParams.get("path") || "";

        const file = getFile(path);
        if (!file || file.type !== "file") {
            return HttpResponse.json(
                { message: "File not found" },
                { status: 404 }
            );
        }

        const lowerPath = path.toLowerCase();
        const isImage =
            lowerPath.endsWith(".png") || lowerPath.endsWith(".jpg") || lowerPath.endsWith(".jpeg");

        return new HttpResponse(new Blob([file.content || ""]), {
            headers: {
                "Content-Type": isImage ? "image/png" : "application/octet-stream",
            },
        });
    }),

    // POST /api/workspace/open - Set workspace path
    http.post(`${API_BASE}/workspace/open`, async ({ request }) => {
        const body = (await request.json()) as { path: string; create_if_missing?: boolean };

        // Mock implementation - just return workspace info
        const projects = getAllProjects();
        const assets = getAllAssets();

        return HttpResponse.json({
            root: body.path,
            projectCount: projects.length,
            assetCount: assets.length,
        });
    }),

    // POST /api/workspace/directories - Create directory
    http.post(`${API_BASE}/workspace/directories`, async ({ request }) => {
        const body = (await request.json()) as { folder_id: string; path: string };

        createDirectory(body.path);

        return HttpResponse.json({
            path: body.path,
        });
    }),

    // PUT /api/workspace/files - Write file content
    http.put(`${API_BASE}/workspace/files`, async ({ request }) => {
        const body = (await request.json()) as {
            folder_id: string;
            path: string;
            content: string;
        };

        writeFile(body.path, body.content);

        return HttpResponse.json({
            path: body.path,
        });
    }),

    // DELETE /api/workspace/files - Optional file deletion used by explorer actions
    http.delete(`${API_BASE}/workspace/files`, ({ request }) => {
        const url = new URL(request.url);
        const path = url.searchParams.get("path") || "";
        const deleted = deleteFile(path);

        if (!deleted) {
            return HttpResponse.json({ message: "File not found" }, { status: 404 });
        }

        return HttpResponse.json({ path });
    }),
];
