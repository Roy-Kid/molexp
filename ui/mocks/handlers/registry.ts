/**
 * Mock handlers for Registry API endpoints
 * 
 * Note: These endpoints were not found in the backend routes but are included
 * for potential future use based on the proposal requirements.
 */

import { http, HttpResponse } from "msw";

const API_BASE = "/api";

export const registryHandlers = [
    // GET /api/registry/tasks - List available task types (placeholder)
    http.get(`${API_BASE}/registry/tasks`, () => {
        // Mock task registry
        const tasks = [
            {
                id: "train",
                label: "Train Model",
                category: "ml",
                description: "Train a machine learning model",
                inputs: [
                    { id: "dataset", label: "Dataset", type: "asset", required: true },
                    { id: "config", label: "Config", type: "json", required: false },
                ],
                outputs: [
                    { id: "model", label: "Model", type: "asset" },
                ],
            },
            {
                id: "evaluate",
                label: "Evaluate Model",
                category: "ml",
                description: "Evaluate model performance",
                inputs: [
                    { id: "model", label: "Model", type: "asset", required: true },
                    { id: "test_data", label: "Test Data", type: "asset", required: true },
                ],
                outputs: [
                    { id: "metrics", label: "Metrics", type: "json" },
                ],
            },
        ];

        return HttpResponse.json({ tasks });
    }),
];
