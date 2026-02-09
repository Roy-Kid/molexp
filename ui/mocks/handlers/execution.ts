/**
 * Mock handlers for Execution API endpoints
 * 
 * Note: These endpoints were not found in the backend routes but are included
 * for potential future use based on the proposal requirements.
 */

import { http, HttpResponse } from "msw";
import { setRunStatus, addRunLog } from "../db";

const API_BASE = "/api";

export const executionHandlers = [
    // POST /api/execute - Trigger run execution (placeholder)
    http.post(`${API_BASE}/execute`, async () => {

        // Mock run ID
        const runId = `run-${Date.now()}`;

        // Simulate async execution
        setTimeout(() => {
            setRunStatus(runId, "running");
            addRunLog(runId, "Starting execution...");

            setTimeout(() => {
                setRunStatus(runId, "succeeded");
                addRunLog(runId, "Execution completed successfully");
            }, 2000);
        }, 100);

        return HttpResponse.json({
            runId,
            status: "pending",
        }, { status: 202 });
    }),

    // GET /api/runs/:id/status - Poll run status (placeholder)
    http.get(`${API_BASE}/runs/:id/status`, ({ params }) => {
        const { id } = params;

        // Mock status response
        return HttpResponse.json({
            id,
            status: "running",
            progress: 0.5,
        });
    }),
];
