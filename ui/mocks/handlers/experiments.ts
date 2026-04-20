/**
 * Mock handlers for Experiment API endpoints
 */

import { http, HttpResponse } from "msw";
import { getExperimentsByProject, setExperiment, deleteExperiment } from "../db";
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
