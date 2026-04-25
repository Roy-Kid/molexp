/**
 * Mock handlers for catalog reverse-lookup endpoints.
 *
 * Implements `GET /api/catalog/by-path` so the unified file tree can ask
 * "who produced this file" for any node — without a live backend.
 */

import { http, HttpResponse } from "msw";

const API_BASE = "/api";

interface PathScope {
    kind: "workspace" | "project" | "experiment" | "run";
    projectId: string | null;
    experimentId: string | null;
    runId: string | null;
}

function deriveScopeFromPath(rel: string): PathScope {
    const parts = rel.split("/").filter(Boolean);
    if (parts[0] !== "projects") {
        return { kind: "workspace", projectId: null, experimentId: null, runId: null };
    }

    const projectId = parts[1] ?? null;
    let experimentId: string | null = null;
    let runId: string | null = null;
    if (parts[2] === "experiments" && parts[3]) experimentId = parts[3];
    if (parts[4] === "runs" && parts[5]) {
        runId = parts[5].startsWith("run-") ? parts[5].slice(4) : parts[5];
    }

    let kind: PathScope["kind"] = "workspace";
    if (runId) kind = "run";
    else if (experimentId) kind = "experiment";
    else if (projectId) kind = "project";

    return { kind, projectId, experimentId, runId };
}

function isUnderArtifacts(rel: string): boolean {
    return rel.includes("/artifacts/") || rel.includes("/logs/") || rel.includes("/checkpoints/");
}

export const catalogHandlers = [
    // GET /api/catalog/by-path
    http.get(`${API_BASE}/catalog/by-path`, ({ request }) => {
        const url = new URL(request.url);
        const rel = (url.searchParams.get("path") ?? "").replace(/^\/+/, "");
        const scope = deriveScopeFromPath(rel);

        // Pretend any path inside artifacts/logs/checkpoints is a registered asset.
        const matched = isUnderArtifacts(rel);
        const segments = rel.split("/");
        const taskId = scope.runId ? "train" : null;
        const assetKind = rel.includes("/logs/")
            ? "log"
            : rel.includes("/checkpoints/")
                ? "checkpoint"
                : "artifact";

        const siblings = matched
            ? [
                  {
                      assetId: `${scope.runId}-sibling-1`,
                      name: "metrics.json",
                      kind: "artifact",
                      relPath: "artifacts/metrics.json",
                  },
              ]
            : [];

        return HttpResponse.json({
            matched,
            workspaceRelPath: rel,
            assetId: matched ? `mock-${segments[segments.length - 1]}` : null,
            assetKind: matched ? assetKind : null,
            producer: matched
                ? { runId: scope.runId, taskId, executionId: null }
                : null,
            scope,
            siblings,
        });
    }),
];
