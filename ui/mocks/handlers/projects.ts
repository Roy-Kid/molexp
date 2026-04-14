/**
 * Mock handlers for Project API endpoints
 */

import { http, HttpResponse } from "msw";
import { getAllProjects, getProject, setProject, deleteProject, getAssetsByProject } from "../db";
import type { ApiProjectResponse } from "../../src/app/types";
import type { ProjectCreateRequest } from "../../src/api/generated/models/ProjectCreateRequest";

const API_BASE = "/api";

export const projectHandlers = [
    // GET /api/projects - List all projects
    http.get(`${API_BASE}/projects`, () => {
        const projects = getAllProjects();
        return HttpResponse.json(projects);
    }),

    // GET /api/projects - List all projects
    http.get(`${API_BASE}/projects/:id`, ({ params }) => {
        const { id } = params;
        const project = getProject(id as string);

        if (!project) {
            return HttpResponse.json(
                { message: `Project ${id} not found` },
                { status: 404 }
            );
        }

        return HttpResponse.json(project);
    }),

    // GET /api/projects/:id/assets - Get project assets
    http.get(`${API_BASE}/projects/:id/assets`, ({ params }) => {
        const assets = getAssetsByProject(params.id as string);
        return HttpResponse.json(assets);
    }),

    // POST /api/projects - Create new project
    http.post(`${API_BASE}/projects`, async ({ request }) => {
        const body = (await request.json()) as ProjectCreateRequest;

        const newProject: ApiProjectResponse = {
            id: body.name?.toLowerCase().replace(/\s+/g, "-") || `project-${Date.now()}`,
            projectId: body.name?.toLowerCase().replace(/\s+/g, "-") || `project-${Date.now()}`,
            name: body.name || "New Project",
            description: body.description || "",
            owner: body.owner || "molexp",
            tags: body.tags || [],
            config: {},
            created: new Date().toISOString(),
            experimentCount: 0,
        };

        setProject(newProject);

        return HttpResponse.json(newProject, { status: 201 });
    }),

    // DELETE /api/projects/:id - Delete project
    http.delete(`${API_BASE}/projects/:id`, ({ params }) => {
        const { id } = params;
        const deleted = deleteProject(id as string);

        if (!deleted) {
            return HttpResponse.json(
                { message: `Project ${id} not found` },
                { status: 404 }
            );
        }

        return HttpResponse.json({ message: "Project deleted" });
    }),
];
