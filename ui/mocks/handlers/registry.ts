/**
 * Mock handlers for Registry API endpoints
 */

import { http, HttpResponse } from "msw";

const API_BASE = "/api";

export const registryHandlers = [
  http.get(`${API_BASE}/plugins`, () => {
    return HttpResponse.json({
      plugins: [
        {
          id: "core",
          title: "Core Workspace UI",
          description: "Built-in Molexp workspace renderers and previews.",
          uiModule: "core",
          capabilities: ["workspace", "renderers", "file_previews"],
          metadata: {},
        },
        {
          id: "molq",
          title: "Molq",
          description: "Scheduler-aware run viewers and monitor surfaces for molq-backed runs.",
          uiModule: "molq",
          capabilities: ["submit", "monitor", "scheduler_inspector"],
          metadata: {
            schedulers: ["local", "slurm"],
          },
        },
      ],
      total: 2,
    });
  }),

  http.get(`${API_BASE}/tasks`, () => {
    return HttpResponse.json(
      {
        error: "not_implemented",
        message: "Task registry is being re-implemented in Phase 3.",
        tasks: [],
      },
      { status: 501 },
    );
  }),

  http.get(`${API_BASE}/tasks/:nodeId`, ({ params }) => {
    return HttpResponse.json(
      {
        error: "not_implemented",
        message: `Task registry is being re-implemented in Phase 3. Task '${params.nodeId}' not found.`,
      },
      { status: 501 },
    );
  }),
];
