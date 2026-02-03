/**
 * Mock handlers for Run API endpoints
 */

import { http, HttpResponse } from "msw";
import { getRunsByExperiment, setRun, deleteRun } from "../db";
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

            const newRun: ApiRunResponse = {
                id: body.git_commit ? `run-${body.git_commit.substring(0, 7)}` : `run-${Date.now()}`,
                runId: body.git_commit ? `run-${body.git_commit.substring(0, 7)}` : `run-${Date.now()}`,
                projectId: projectId as string,
                experimentId: experimentId as string,
                status: "pending",
                finished: null,
                parameters: body.parameters || {},
                created: new Date().toISOString(),
                workflow: {
                    file: body.workflow_file,
                    gitCommit: body.git_commit || null,
                    serializedGraph: null
                },
                executorInfo: {},
                workingDir: null,
                logsDir: null,
                assetRefs: { inputs: [], outputs: [] },
                context: null,
            };

            setRun(newRun);
            return HttpResponse.json(newRun, { status: 201 });
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
];
