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
            plan_mode?: boolean;
            instructions_override?: string | null;
            skill_id?: string | null;
        };

        const sessionId = `sess-${Date.now()}`;
        const startedAt = new Date().toISOString();
        const planMode = Boolean(body.plan_mode);

        const newSession: ApiAgentSession = {
            sessionId,
            status: "running",
            goalDescription: body.description,
            createdAt: startedAt,
            events: [
                {
                    type: "PlanCreatedEvent",
                    ts: startedAt,
                    payload: {
                        plan_steps: [
                            "Analyse workspace and identify relevant experiments",
                            "Execute requested workflow steps",
                            "Summarise results and produced assets",
                        ],
                    },
                },
            ],
            stats: {
                inputTokens: 1280,
                outputTokens: 320,
                cacheReadTokens: 512,
                cacheWriteTokens: 0,
                totalTokens: 1600,
                requests: 1,
                toolCalls: 0,
                events: 1,
                startedAt,
                completedAt: null,
                durationSeconds: null,
            },
            planMode,
            skillId: body.skill_id ?? null,
        };

        setAgentSession(newSession);

        // Simulate session completing after a short delay (mock only)
        setTimeout(() => {
            const session = getAgentSession(sessionId);
            if (!session) return;
            const completedAt = new Date().toISOString();
            const planSummary = [
                "1. list_projects() — inspect available projects",
                "2. list_experiments(project_id=p1) — identify the relevant ablation",
                "3. get_run_results(...) — pull the metric series for plotting",
                "4. run_python(code=...) — produce a Plotly scatter chart",
            ].join("\n");
            const sampleArtifact = {
                type: "ResultArtifactEvent",
                ts: completedAt,
                payload: {
                    kind: "plot",
                    title: "Sample: total_energy vs temperature",
                    payload: {
                        data: [
                            {
                                type: "scatter",
                                mode: "lines+markers",
                                x: [200, 300, 400, 500, 600],
                                y: [-10.4, -10.1, -9.8, -9.4, -9.0],
                                name: "energy",
                            },
                        ],
                        layout: {
                            xaxis: { title: "temperature (K)" },
                            yaxis: { title: "total_energy (eV)" },
                        },
                    },
                },
            };
            const eventsTail = planMode
                ? [
                      {
                          type: "SessionCompletedEvent",
                          ts: completedAt,
                          payload: {
                              summary: planSummary,
                              produced_runs: [],
                          },
                      },
                  ]
                : [
                      {
                          type: "ObservationEvent",
                          ts: completedAt,
                          payload: { content: "All steps completed successfully." },
                      },
                      sampleArtifact,
                      {
                          type: "SessionCompletedEvent",
                          ts: completedAt,
                          payload: {
                              summary: `Goal completed: "${body.description}"`,
                              produced_runs: [],
                          },
                      },
                  ];
            const completed: ApiAgentSession = {
                ...session,
                status: "completed",
                events: [...(session.events ?? []), ...eventsTail],
                stats: {
                    inputTokens: 4860,
                    outputTokens: 1240,
                    cacheReadTokens: 2048,
                    cacheWriteTokens: 256,
                    totalTokens: 6100,
                    requests: 4,
                    toolCalls: planMode ? 1 : 3,
                    events: planMode ? 2 : 3,
                    startedAt,
                    completedAt,
                    durationSeconds:
                        (new Date(completedAt).getTime() - new Date(startedAt).getTime()) / 1000,
                },
            };
            setAgentSession(completed);
        }, planMode ? 1500 : 3000);

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
                for (const event of session.events ?? []) {
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

    // POST /api/agent/sessions/:sessionId/execute-plan — promote plan-mode session
    http.post(`${API_BASE}/agent/sessions/:sessionId/execute-plan`, ({ params }) => {
        const original = getAgentSession(params.sessionId as string);
        if (!original) {
            return HttpResponse.json(
                { detail: `Session ${params.sessionId} not found` },
                { status: 404 },
            );
        }
        if (!original.planMode) {
            return HttpResponse.json(
                { detail: "Session was not started in plan mode." },
                { status: 409 },
            );
        }
        const newId = `sess-${Date.now()}`;
        const startedAt = new Date().toISOString();
        const followUp: ApiAgentSession = {
            sessionId: newId,
            status: "running",
            goalDescription: original.goalDescription,
            createdAt: startedAt,
            events: [],
            stats: {
                inputTokens: 0,
                outputTokens: 0,
                cacheReadTokens: 0,
                cacheWriteTokens: 0,
                totalTokens: 0,
                requests: 0,
                toolCalls: 0,
                events: 0,
                startedAt,
                completedAt: null,
                durationSeconds: null,
            },
            planMode: false,
            skillId: original.skillId ?? null,
        };
        setAgentSession(followUp);
        return HttpResponse.json(followUp);
    }),

    // GET /api/agent/sessions/:sessionId/system-prompt
    http.get(`${API_BASE}/agent/sessions/:sessionId/system-prompt`, ({ params }) => {
        const session = getAgentSession(params.sessionId as string);
        if (!session) {
            return HttpResponse.json(
                { detail: `Session ${params.sessionId} not found` },
                { status: 404 },
            );
        }
        const base =
            "You are a research experiment assistant integrated with the molexp workspace.";
        const planAddendum =
            "\n\nYou are in PLAN MODE.\nTools that mutate workspace state are unavailable in this turn.";
        return HttpResponse.json({
            base,
            workspaceInstructions: "",
            skillInstructions: "",
            sessionOverride: null,
            planMode: Boolean(session.planMode),
            effective: session.planMode ? base + planAddendum : base,
        });
    }),

    // POST /api/agent/sessions/:sessionId/messages — chat message from user
    http.post(`${API_BASE}/agent/sessions/:sessionId/messages`, async ({ params, request }) => {
        const sessionId = params.sessionId as string;
        const session = getAgentSession(sessionId);
        if (!session) {
            return HttpResponse.json(
                { detail: `Session ${sessionId} not found` },
                { status: 404 },
            );
        }
        const body = (await request.json()) as { content: string; request_id?: string | null };
        const ts = new Date().toISOString();
        const updated: ApiAgentSession = {
            ...session,
            events: [
                ...(session.events ?? []),
                {
                    type: "UserMessageEvent",
                    ts,
                    payload: { content: body.content, request_id: body.request_id ?? null },
                },
            ],
        };
        setAgentSession(updated);

        // After 500 ms, append a synthetic assistant observation acknowledging the message.
        setTimeout(() => {
            const current = getAgentSession(sessionId);
            if (!current) return;
            const reply: ApiAgentSession = {
                ...current,
                events: [
                    ...(current.events ?? []),
                    {
                        type: "ObservationEvent",
                        ts: new Date().toISOString(),
                        payload: { content: `Got it: "${body.content}". Continuing…` },
                    },
                ],
            };
            setAgentSession(reply);
        }, 500);

        return HttpResponse.json({ message: "queued" });
    }),
];
