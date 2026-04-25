/**
 * Mock handlers for Run API endpoints
 */

import { http, HttpResponse } from "msw";
import { getRun, getRunsByExperiment, setRun, deleteRun } from "../db";
import type { ApiRunResponse } from "../../src/app/types";
import type { RunCreateRequest } from "../../src/api/generated/models/RunCreateRequest";

const API_BASE = "/api";

export const runHandlers = [
    // GET /api/projects/:projectId/experiments/:experimentId/runs - List runs for an experiment
    http.get(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs`,
        ({ params }) => {
            const { experimentId } = params;
            const runs = getRunsByExperiment(experimentId as string);
            return HttpResponse.json(runs);
        }
    ),
    // POST /api/projects/:projectId/experiments/:experimentId/runs - Create new run
    http.post(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs`,
        async ({ request, params }) => {
            const { projectId, experimentId } = params;
            const body = (await request.json()) as RunCreateRequest;
            const runId = `run-${Date.now()}`;

            const newRun: ApiRunResponse = {
                id: runId,
                projectId: projectId as string,
                experimentId: experimentId as string,
                status: "pending",
                finished: null,
                parameters: body.parameters || {},
                created: new Date().toISOString(),
                executorInfo: {},
            };

            setRun(newRun);
            return HttpResponse.json(newRun, { status: 201 });
        }
    ),

    // GET /api/projects/:projectId/experiments/:experimentId/runs/:runId - Run detail
    http.get(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId`,
        ({ params }) => {
            const run = getRun(params.runId as string);
            if (!run) {
                return HttpResponse.json({ detail: "Run not found" }, { status: 404 });
            }

            return HttpResponse.json(run);
        }
    ),

    // GET /api/projects/:projectId/experiments/:experimentId/runs/:runId/metrics - Run metrics
    http.get(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/metrics`,
        ({ request }) => {
            const url = new URL(request.url);
            const sinceLine = Number(url.searchParams.get("since_line") ?? "0");
            const allRecords = [
                { t: "scalar", k: "train/loss", s: 1, w: new Date().toISOString(), v: 0.31 },
                { t: "scalar", k: "train/loss", s: 2, w: new Date().toISOString(), v: 0.24 },
                { t: "scalar", k: "eval/acc", s: 2, w: new Date().toISOString(), v: 0.82 },
            ];
            const records = allRecords.slice(sinceLine);

            return HttpResponse.json({
                nextLine: allRecords.length,
                records,
                series: [
                    {
                        key: "eval/acc",
                        type: "scalar",
                        count: 1,
                        latestStep: 2,
                        latestTimestamp: allRecords[2].w,
                        latestValue: 0.82,
                    },
                    {
                        key: "train/loss",
                        type: "scalar",
                        count: 2,
                        latestStep: 2,
                        latestTimestamp: allRecords[1].w,
                        latestValue: 0.24,
                    },
                ],
                parseErrors: 0,
            });
        }
    ),

    // PATCH /api/projects/:projectId/experiments/:experimentId/runs/:runId/status - Update status
    http.patch(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/status`,
        async ({ params, request }) => {
            const run = getRun(params.runId as string);
            if (!run) {
                return HttpResponse.json({ detail: "Run not found" }, { status: 404 });
            }

            const body = (await request.json()) as { status?: string };
            const nextStatus = body.status ?? run.status;
            const updated: ApiRunResponse = {
                ...run,
                status: nextStatus,
                finished:
                    nextStatus === "succeeded" || nextStatus === "failed" || nextStatus === "cancelled"
                        ? new Date().toISOString()
                        : run.finished ?? null,
            };

            setRun(updated);

            return HttpResponse.json({
                id: updated.id,
                status: updated.status,
                finished: updated.finished,
            });
        }
    ),

    // POST /api/projects/:projectId/experiments/:experimentId/runs/:runId/start - Start run
    http.post(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/start`,
        ({ params }) => {
            const run = getRun(params.runId as string);
            if (!run) {
                return HttpResponse.json({ detail: "Run not found" }, { status: 404 });
            }

            const updated: ApiRunResponse = {
                ...run,
                status: "running",
            };
            setRun(updated);
            return HttpResponse.json(updated);
        }
    ),

    // DELETE /api/projects/:projectId/experiments/:experimentId/runs/:runId
    http.delete(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId`,
        ({ params }) => {
            const { runId } = params;
            const deleted = deleteRun(runId as string);

            if (!deleted) {
                return HttpResponse.json(
                    { message: `Run ${runId} not found` },
                    { status: 404 }
                );
            }

            return HttpResponse.json({ message: "Run deleted" });
        }
    ),

    // GET /api/projects/:projectId/experiments/:experimentId/runs/:runId/files - Output file tree
    http.get(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/files`,
        ({ params }) => {
            const run = getRun(params.runId as string);
            if (!run) {
                return HttpResponse.json({ detail: "Run not found" }, { status: 404 });
            }
            const now = Date.now() / 1000;
            return HttpResponse.json({
                runId: run.id,
                runDir: `projects/${params.projectId}/experiments/${params.experimentId}/runs/run-${run.id}`,
                nodes: [
                    {
                        name: "artifacts",
                        relPath: "artifacts",
                        type: "folder",
                        modified: now,
                        children: [
                            {
                                name: "model.bin",
                                relPath: "artifacts/model.bin",
                                type: "file",
                                size: 4096,
                                modified: now,
                                assetId: `${run.id}-model`,
                                assetKind: "artifact",
                                taskId: "train",
                                children: [],
                            },
                            {
                                name: "metrics.json",
                                relPath: "artifacts/metrics.json",
                                type: "file",
                                size: 192,
                                modified: now,
                                assetId: `${run.id}-metrics`,
                                assetKind: "artifact",
                                taskId: "train",
                                children: [],
                            },
                        ],
                    },
                    {
                        name: "logs",
                        relPath: "logs",
                        type: "folder",
                        modified: now,
                        children: [
                            {
                                name: "run.log",
                                relPath: "logs/run.log",
                                type: "file",
                                size: 1024,
                                modified: now,
                                assetId: `${run.id}-runlog`,
                                assetKind: "log",
                                taskId: null,
                                children: [],
                            },
                        ],
                    },
                ],
            });
        }
    ),

    // POST /api/projects/:projectId/experiments/:experimentId/runs/:runId/rerun
    http.post(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/rerun`,
        ({ params }) => {
            const run = getRun(params.runId as string);
            if (!run) {
                return HttpResponse.json({ detail: "Run not found" }, { status: 404 });
            }
            const newRunId = `run-${Date.now()}`;
            const cloned: ApiRunResponse = {
                id: newRunId,
                projectId: run.projectId,
                experimentId: run.experimentId,
                status: "pending",
                finished: null,
                parameters: { ...run.parameters },
                created: new Date().toISOString(),
                executorInfo: {},
            };
            setRun(cloned);
            return HttpResponse.json(
                {
                    sourceRunId: run.id,
                    newRunId,
                    projectId: run.projectId,
                    experimentId: run.experimentId,
                    status: cloned.status,
                },
                { status: 201 }
            );
        }
    ),

    // POST /api/projects/:projectId/experiments/:experimentId/runs/:runId/kill
    http.post(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/kill`,
        ({ params }) => {
            const run = getRun(params.runId as string);
            if (!run) {
                return HttpResponse.json({ detail: "Run not found" }, { status: 404 });
            }
            const updated: ApiRunResponse = {
                ...run,
                status: "cancelled",
                finished: new Date().toISOString(),
            };
            setRun(updated);
            return HttpResponse.json({
                runId: run.id,
                status: "cancelled",
                message: "Run marked as cancelled",
            });
        }
    ),

    // GET /api/projects/:projectId/experiments/:experimentId/runs/:runId/export
    http.get(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/export`,
        ({ params }) => {
            const run = getRun(params.runId as string);
            if (!run) {
                return HttpResponse.json({ detail: "Run not found" }, { status: 404 });
            }
            // Mock zip — not a valid archive, but enough to round-trip a Blob.
            return new HttpResponse(new Blob(["MOCK_RUN_EXPORT_ZIP"]), {
                headers: {
                    "Content-Type": "application/zip",
                    "Content-Disposition": `attachment; filename="run-${run.id}.zip"`,
                },
            });
        }
    ),
];
