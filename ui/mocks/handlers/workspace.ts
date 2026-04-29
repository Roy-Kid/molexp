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
 * Build a synthetic LAMMPS log for mock mode so the molvis plugin can parse
 * thermo output without needing a real run on disk. Used by the workspace
 * file-text endpoint; the ``/lammps-log`` route mocks pre-parsed thermo
 * directly (see runs.ts).
 */
function synthesizeLammpsLog(): string {
    const STEPS = 50;
    const lines: string[] = [
        "LAMMPS (29 Sep 2021)",
        "Step Temp PotEng KinEng TotEng Press",
    ];
    for (let i = 0; i <= STEPS; i += 1) {
        const t = i / STEPS;
        const step = i * 1000;
        const temp = 300 - 100 * Math.exp(-3 * t) + (Math.random() - 0.5) * 5;
        const peng = -1500 + 600 * Math.exp(-2 * t) + (Math.random() - 0.5) * 8;
        const keng = (3 / 2) * temp;
        const teng = peng + keng;
        const press = 1.0 - 0.4 * Math.exp(-2.5 * t) + (Math.random() - 0.5) * 0.05;
        lines.push(
            `${step} ${temp.toFixed(4)} ${peng.toFixed(4)} ${keng.toFixed(4)} ${teng.toFixed(4)} ${press.toFixed(4)}`,
        );
    }
    lines.push(`Loop time of 1.234 on 1 procs for ${STEPS * 1000} steps with 1000 atoms`);
    return lines.join("\n");
}

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

interface MockExecutionRow {
    executionId: string;
    runId: string;
    status: string;
    startedAt: string;
    finishedAt: string | null;
    durationSeconds: number | null;
    schedulerJobId: string | null;
    backend: string | null;
    metadata: Record<string, string>;
    backendMetadata: Record<string, string>;
}

interface MockRunRow {
    id: string;
    name: string;
    projectId: string;
    projectName: string;
    experimentId: string;
    experimentName: string;
    status: string;
    backend: string | null;
    cluster: string | null;
    scheduler: string | null;
    profile: string | null;
    parameters: Record<string, unknown>;
    createdAt: string;
    finishedAt: string | null;
    executionCount: number;
    latestSchedulerJobId: string | null;
    executions: MockExecutionRow[];
}

const baseSubmittedAt = Date.now() - 6 * 3600 * 1000;

function buildMockWorkspaceRuns(): MockRunRow[] {
    const fixtures: Array<{
        id: string;
        projectId: string;
        projectName: string;
        experimentId: string;
        experimentName: string;
        status: string;
        backend: string | null;
        cluster: string | null;
        scheduler: string | null;
        profile: string | null;
        attempts: Array<{ status: string; schedulerJobId?: string; durationSec?: number }>;
    }> = [
        {
            id: "run-allegro-001",
            projectId: "proj-allegro",
            projectName: "Allegro Sweep",
            experimentId: "exp-lr-grid",
            experimentName: "Learning rate grid",
            status: "running",
            backend: "molq",
            cluster: "dardel.scilifelab.se",
            scheduler: "slurm",
            profile: "dardel-gpu",
            attempts: [
                { status: "failed", schedulerJobId: "48201", durationSec: 1810 },
                { status: "running", schedulerJobId: "48317", durationSec: 4302 },
            ],
        },
        {
            id: "run-nemd-014",
            projectId: "proj-electrolytes",
            projectName: "Electrolyte transport",
            experimentId: "exp-nemd",
            experimentName: "NEMD conductivity",
            status: "succeeded",
            backend: "molq",
            cluster: "alvis.scilifelab.se",
            scheduler: "slurm",
            profile: "alvis",
            attempts: [
                { status: "succeeded", schedulerJobId: "910023", durationSec: 7200 },
            ],
        },
        {
            id: "run-local-quick",
            projectId: "proj-allegro",
            projectName: "Allegro Sweep",
            experimentId: "exp-smoke",
            experimentName: "Smoke test",
            status: "succeeded",
            backend: "local",
            cluster: null,
            scheduler: "local",
            profile: null,
            attempts: [{ status: "succeeded", durationSec: 35 }],
        },
        {
            id: "run-pending",
            projectId: "proj-electrolytes",
            projectName: "Electrolyte transport",
            experimentId: "exp-nemd",
            experimentName: "NEMD conductivity",
            status: "pending",
            backend: null,
            cluster: null,
            scheduler: null,
            profile: null,
            attempts: [],
        },
    ];

    return fixtures.map((row, runIdx) => {
        const createdAt = new Date(baseSubmittedAt + runIdx * 12 * 60_000).toISOString();
        const executions: MockExecutionRow[] = row.attempts.map((attempt, attemptIdx) => {
            const startedAt = new Date(
                baseSubmittedAt + runIdx * 12 * 60_000 + attemptIdx * 600_000,
            );
            const finishedAt =
                attempt.durationSec && attempt.status !== "running"
                    ? new Date(startedAt.getTime() + attempt.durationSec * 1000)
                    : null;
            const metadata: Record<string, string> = {};
            if (row.cluster) metadata.cluster_name = row.cluster;
            if (row.scheduler) metadata.scheduler = row.scheduler;
            if (attempt.schedulerJobId) metadata.scheduler_job_id = attempt.schedulerJobId;

            return {
                executionId: `exec-${row.id}-${attemptIdx + 1}`,
                runId: row.id,
                status: attempt.status,
                startedAt: startedAt.toISOString(),
                finishedAt: finishedAt?.toISOString() ?? null,
                durationSeconds: attempt.durationSec ?? null,
                schedulerJobId: attempt.schedulerJobId ?? null,
                backend: row.backend,
                metadata,
                backendMetadata: metadata,
            };
        });

        return {
            ...row,
            name: row.id,
            parameters: { lr: 0.0001, batch: 32 },
            createdAt,
            finishedAt:
                row.status === "succeeded" || row.status === "failed"
                    ? executions[executions.length - 1]?.finishedAt ?? null
                    : null,
            executionCount: executions.length,
            latestSchedulerJobId:
                executions
                    .slice()
                    .reverse()
                    .find((e) => e.schedulerJobId)?.schedulerJobId ?? null,
            executions,
        };
    });
}

function computeMockStats(rows: MockRunRow[]): {
    total: number;
    running: number;
    pending: number;
    failed: number;
    succeeded: number;
} {
    const stats = { total: rows.length, running: 0, pending: 0, failed: 0, succeeded: 0 };
    for (const row of rows) {
        switch (row.status) {
            case "running":
                stats.running += 1;
                break;
            case "pending":
            case "queued":
            case "submitted":
            case "created":
                stats.pending += 1;
                break;
            case "failed":
            case "cancelled":
            case "timed_out":
            case "lost":
                stats.failed += 1;
                break;
            case "succeeded":
                stats.succeeded += 1;
                break;
        }
    }
    return stats;
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

    // GET /api/workspace/runs - Cross-experiment runs aggregator
    http.get(`${API_BASE}/workspace/runs`, ({ request }) => {
        const url = new URL(request.url);
        const projectFilter = url.searchParams.get("projectId");
        const experimentFilter = url.searchParams.get("experimentId");
        const backendFilter = url.searchParams.get("backend");
        const statusFilter = url.searchParams.get("status");

        const allRuns = buildMockWorkspaceRuns();
        const filtered = allRuns.filter((row) => {
            if (projectFilter && row.projectId !== projectFilter) return false;
            if (experimentFilter && row.experimentId !== experimentFilter) return false;
            if (backendFilter && (row.backend ?? "") !== backendFilter) return false;
            if (statusFilter && row.status !== statusFilter) return false;
            return true;
        });

        return HttpResponse.json({
            runs: filtered,
            stats: computeMockStats(filtered),
            total: filtered.length,
            truncated: false,
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

        if (path.endsWith("log.lammps") || path.endsWith(".lammps.log")) {
            return HttpResponse.json({ content: synthesizeLammpsLog() });
        }

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
