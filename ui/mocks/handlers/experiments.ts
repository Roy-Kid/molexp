/**
 * Mock handlers for Experiment API endpoints
 */

import { http, HttpResponse } from "msw";
import {
    deleteExperiment,
    getExperimentsByProject,
    getRunsByExperiment,
    setExperiment,
} from "../db";
import type { ApiExperimentResponse } from "../../src/app/types";
import type { ExperimentCreateRequest } from "../../src/api/generated/models/ExperimentCreateRequest";

const API_BASE = "/api";

export const experimentHandlers = [
    // GET /api/projects/:projectId/experiments - List experiments for a project
    http.get(`${API_BASE}/projects/:projectId/experiments`, ({ params }) => {
        const { projectId } = params;
        const experiments = getExperimentsByProject(projectId as string);
        return HttpResponse.json(experiments);
    }),
    // POST /api/projects/:projectId/experiments - Create new experiment
    http.post(`${API_BASE}/projects/:projectId/experiments`, async ({ request, params }) => {
        const { projectId } = params;
        const body = (await request.json()) as ExperimentCreateRequest;

        const experimentId =
            body.name?.toLowerCase().replace(/\s+/g, "-") || `exp-${Date.now()}`;
        const newExperiment: ApiExperimentResponse = {
            id: experimentId,
            projectId: projectId as string,
            name: body.name,
            description: body.description || "",
            workflow: body.workflow_source ?? null,
            workflowType: "yaml",
            gitCommit: null,
            parameterSpace: body.parameter_space || {},
            runCount: 0,
            runs: [],
            created: new Date().toISOString(),
        };

        setExperiment(newExperiment);
        return HttpResponse.json(newExperiment, { status: 201 });
    }),

    // GET /api/projects/:projectId/experiments/:experimentId/comparison - Sweep matrix
    http.get(
        `${API_BASE}/projects/:projectId/experiments/:experimentId/comparison`,
        ({ params }) => {
            const { projectId, experimentId } = params;
            const runs = getRunsByExperiment(experimentId as string);
            const paramKeys = new Set<string>();
            for (const r of runs) {
                Object.keys(r.parameters ?? {}).forEach((k) => paramKeys.add(k));
            }
            const metricKeys = ["train/loss", "eval/acc"];
            return HttpResponse.json({
                experimentId: experimentId as string,
                projectId: projectId as string,
                paramKeys: Array.from(paramKeys).sort(),
                metricKeys,
                runs: runs.map((r) => {
                    const finished = r.finished ? new Date(r.finished).getTime() : null;
                    const created = new Date(r.created).getTime();
                    const durationSec =
                        finished !== null ? Math.max(0, (finished - created) / 1000) : null;
                    return {
                        runId: r.id,
                        status: r.status,
                        parameters: r.parameters,
                        metrics: { "train/loss": 0.18, "eval/acc": 0.84 },
                        durationSec,
                        created: r.created,
                        finished: r.finished ?? null,
                        error: null,
                    };
                }),
            });
        }
    ),

    // DELETE /api/projects/:projectId/experiments/:experimentId - Delete experiment
    http.delete(`${API_BASE}/projects/:projectId/experiments/:experimentId`, ({ params }) => {
        const { experimentId } = params;
        const deleted = deleteExperiment(experimentId as string);

        if (!deleted) {
            return HttpResponse.json(
                { message: `Experiment ${experimentId} not found` },
                { status: 404 }
            );
        }

        return HttpResponse.json({ message: "Experiment deleted" });
    }),
];
