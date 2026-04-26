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
            const STEPS = 120;
            const baseWall = Date.now() - STEPS * 1000;
            const allRecords: Array<{
                t: string;
                k: string;
                s: number;
                w: string;
                v: number;
            }> = [];
            for (let step = 1; step <= STEPS; step += 1) {
                const t = step / STEPS;
                const wall = new Date(baseWall + step * 1000).toISOString();
                // Train loss: noisy decay
                const trainLoss =
                    0.05 + 0.6 * Math.exp(-3 * t) + (Math.random() - 0.5) * 0.08;
                // Val loss: similar but plateaus higher with more noise
                const valLoss =
                    0.12 + 0.5 * Math.exp(-2.5 * t) + (Math.random() - 0.5) * 0.12;
                // Train acc: noisy ramp
                const trainAcc =
                    1 - 0.5 * Math.exp(-3 * t) + (Math.random() - 0.5) * 0.05;
                // Val acc: noisier ramp
                const valAcc =
                    1 - 0.55 * Math.exp(-2.2 * t) + (Math.random() - 0.5) * 0.07;
                allRecords.push(
                    { t: "scalar", k: "loss/train", s: step, w: wall, v: trainLoss },
                    { t: "scalar", k: "loss/val", s: step, w: wall, v: valLoss },
                    { t: "scalar", k: "acc/train", s: step, w: wall, v: trainAcc },
                    { t: "scalar", k: "acc/val", s: step, w: wall, v: valAcc },
                );
            }
            const records = allRecords.slice(sinceLine);

            const lastByKey = new Map<string, { step: number; wall: string; value: number }>();
            for (const rec of allRecords) {
                lastByKey.set(rec.k, { step: rec.s, wall: rec.w, value: rec.v });
            }
            const series = Array.from(lastByKey.entries())
                .map(([key, last]) => ({
                    key,
                    type: "scalar",
                    count: STEPS,
                    latestStep: last.step,
                    latestTimestamp: last.wall,
                    latestValue: last.value,
                }))
                .sort((a, b) => a.key.localeCompare(b.key));

            return HttpResponse.json({
                nextLine: allRecords.length,
                records,
                series,
                parseErrors: 0,
            });
        }
    ),

    // GET /api/projects/:projectId/experiments/:experimentId/runs/:runId/lammps-log - LAMMPS thermo
    http.get(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/lammps-log`,
        ({ request }) => {
            const url = new URL(request.url);
            const path = url.searchParams.get("path") ?? "log.lammps";
            const STEPS = 50;
            const rows: number[][] = [];
            for (let i = 0; i < STEPS; i += 1) {
                const step = i * 100;
                const t = i / STEPS;
                const temp = 300 + 5 * Math.sin(t * Math.PI * 2) + (Math.random() - 0.5) * 1.5;
                const potEng = -1000 - 50 * t + (Math.random() - 0.5) * 8;
                const kinEng = 1.5 * temp + (Math.random() - 0.5) * 2;
                const totEng = potEng + kinEng;
                const press = 1.0 + 0.4 * Math.cos(t * Math.PI * 4) + (Math.random() - 0.5) * 0.05;
                rows.push([step, temp, potEng, kinEng, totEng, press]);
            }
            return HttpResponse.json({
                path,
                version: "LAMMPS (mock build)",
                nStages: 1,
                stages: [
                    {
                        columns: ["Step", "Temp", "PotEng", "KinEng", "TotEng", "Press"],
                        rows,
                    },
                ],
            });
        }
    ),

    // GET /api/projects/:projectId/experiments/:experimentId/runs/:runId/file/text - raw text file
    http.get(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/runs/:runId/file/text`,
        ({ request }) => {
            const url = new URL(request.url);
            const path = url.searchParams.get("path") ?? "";
            // Mock 2-frame XYZ trajectory (small molecule).
            const content =
                "3\nframe 0\nO 0.000 0.000 0.000\nH 0.957 0.000 0.000\nH -0.239 0.927 0.000\n" +
                "3\nframe 1\nO 0.000 0.000 0.000\nH 0.967 0.000 0.000\nH -0.249 0.937 0.000\n";
            return HttpResponse.json({
                path,
                content,
                size: content.length,
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
                        name: "metrics.jsonl",
                        relPath: "metrics.jsonl",
                        type: "file",
                        size: 24576,
                        modified: now,
                        assetId: `${run.id}-metrics-stream`,
                        assetKind: "metrics",
                        taskId: "train",
                        children: [],
                    },
                    {
                        name: "log.lammps",
                        relPath: "log.lammps",
                        type: "file",
                        size: 65536,
                        modified: now,
                        assetId: `${run.id}-lammps-log`,
                        assetKind: "log",
                        taskId: "simulate",
                        children: [],
                    },
                    {
                        name: "trajectory.lammpstrj",
                        relPath: "trajectory.lammpstrj",
                        type: "file",
                        size: 8 * 1024 * 1024,
                        modified: now,
                        assetId: `${run.id}-traj`,
                        assetKind: "trajectory",
                        taskId: "simulate",
                        children: [],
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
