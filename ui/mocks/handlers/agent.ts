/**
 * Mock handlers for Agent API endpoints
 */

import { http, HttpResponse } from "msw";
import { getAllAgentSessions, getAgentSession, setAgentSession } from "../db";
import type { ApiAgentSession } from "../../src/app/types";

const API_BASE = "/api";

export const agentHandlers = [
    // GET /api/agent/sessions — list all sessions
    http.get(`${API_BASE}/agent/sessions`, () => {
        const sessions = getAllAgentSessions();
        return HttpResponse.json({ sessions, total: sessions.length });
    }),

    // POST /api/agent/sessions — create a new session
    http.post(`${API_BASE}/agent/sessions`, async ({ request }) => {
        const body = (await request.json()) as {
            description: string;
            constraints?: Record<string, unknown>;
            success_criteria?: string[];
        };

        const sessionId = `sess-${Date.now()}`;
        const newSession: ApiAgentSession = {
            sessionId,
            status: "running",
            goalDescription: body.description,
            createdAt: new Date().toISOString(),
            events: [
                {
                    type: "PlanCreatedEvent",
                    ts: new Date().toISOString(),
                    payload: {
                        plan_steps: [
                            "Analyse workspace and identify relevant experiments",
                            "Execute requested workflow steps",
                            "Summarise results and produced assets",
                        ],
                    },
                },
            ],
        };

        setAgentSession(newSession);

        // Simulate session completing after a short delay (mock only)
        setTimeout(() => {
            const session = getAgentSession(sessionId);
            if (!session) return;
            const completed: ApiAgentSession = {
                ...session,
                status: "completed",
                events: [
                    ...session.events,
                    {
                        type: "ObservationEvent",
                        ts: new Date().toISOString(),
                        payload: { content: "All steps completed successfully." },
                    },
                    {
                        type: "SessionCompletedEvent",
                        ts: new Date().toISOString(),
                        payload: {
                            summary: `Goal completed: "${body.description}"`,
                            produced_runs: [],
                        },
                    },
                ],
            };
            setAgentSession(completed);
        }, 3000);

        return HttpResponse.json(newSession, { status: 201 });
    }),

    // GET /api/agent/sessions/:sessionId — get a single session
    http.get(`${API_BASE}/agent/sessions/:sessionId`, ({ params }) => {
        const session = getAgentSession(params.sessionId as string);
        if (!session) {
            return HttpResponse.json(
                { detail: `Session ${params.sessionId} not found` },
                { status: 404 },
            );
        }
        return HttpResponse.json(session);
    }),

    // GET /api/agent/sessions/:sessionId/events — SSE stream of events
    http.get(`${API_BASE}/agent/sessions/:sessionId/events`, ({ params }) => {
        const session = getAgentSession(params.sessionId as string);

        const encoder = new TextEncoder();
        const stream = new ReadableStream({
            start(controller) {
                if (!session) {
                    controller.enqueue(encoder.encode('data: {"type":"error","message":"Session not found"}\n\n'));
                    controller.close();
                    return;
                }

                // Send all existing events
                for (const event of session.events) {
                    controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
                }

                // Signal done or waiting
                if (session.status !== "running") {
                    controller.enqueue(encoder.encode('data: {"type":"done"}\n\n'));
                } else {
                    controller.enqueue(encoder.encode('data: {"type":"waiting"}\n\n'));
                }

                controller.close();
            },
        });

        return new HttpResponse(stream, {
            headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        });
    }),

    // POST /api/agent/sessions/:sessionId/approve — respond to approval request
    http.post(`${API_BASE}/agent/sessions/:sessionId/approve`, async ({ params, request }) => {
        const session = getAgentSession(params.sessionId as string);
        if (!session) {
            return HttpResponse.json(
                { detail: `Session ${params.sessionId} not found` },
                { status: 404 },
            );
        }
        const body = (await request.json()) as { request_id: string; approved: boolean };
        return HttpResponse.json({ request_id: body.request_id, approved: body.approved, applied: true });
    }),
];
