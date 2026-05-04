/**
 * Mock handlers for Agent API endpoints
 */

import { http, HttpResponse } from "msw";
import { getAllAgentSessions, getAgentSession, setAgentSession } from "../db";
import type { ApiAgentSession } from "../../src/app/types";

const API_BASE = "/api";

interface ApiAgentTask {
    taskId: string;
    title: string;
    goal: string;
    status: string;
    createdAt: string;
    updatedAt: string | null;
    sessionId: string;
    events: ApiAgentSession["events"];
    stats: ApiAgentSession["stats"];
    planMode: boolean;
    skillId: string | null;
}

interface ApiReviewItem {
    id: string;
    kind: string;
    title: string;
    description: string | null;
    riskLevel: "low" | "medium" | "high";
    status: "pending" | "approved" | "rejected" | "expired";
    targetRef: {
        type: string;
        id: string;
        taskId: string | null;
        sessionId: string | null;
    };
    createdAt: string;
    resolvedAt: string | null;
    resolutionComment: string | null;
    metadata: Record<string, unknown>;
}

const mockReviews: ApiReviewItem[] = [];

const taskFromSession = (session: ApiAgentSession): ApiAgentTask => ({
    taskId: session.taskId ?? session.sessionId,
    title: session.goalDescription.split("\n")[0]?.slice(0, 72) || "Untitled agent task",
    goal: session.goalDescription,
    status: session.status,
    createdAt: session.createdAt,
    updatedAt: session.stats?.completedAt ?? session.stats?.startedAt ?? session.createdAt,
    sessionId: session.sessionId,
    events: session.events ?? [],
    stats: session.stats,
    planMode: Boolean(session.planMode),
    skillId: session.skillId ?? null,
});

const createMockTaskSession = (body: {
    description: string;
    plan_mode?: boolean;
    skill_id?: string | null;
}): ApiAgentSession => {
    const sessionId = `sess-${Date.now()}`;
    const taskId = `task-${Date.now()}`;
    const startedAt = new Date().toISOString();
    const planMode = Boolean(body.plan_mode);
    const session: ApiAgentSession = {
        sessionId,
        taskId,
        status: "running",
        goalDescription: body.description,
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
        planMode,
        skillId: body.skill_id ?? null,
    };
    setAgentSession(session);
    setTimeout(() => {
        const current = getAgentSession(sessionId);
        if (!current) return;
        const ts = new Date().toISOString();
        setAgentSession({
            ...current,
            status: planMode ? "running" : "completed",
            events: [
                ...(current.events ?? []),
                planMode
                    ? {
                          type: "PlanCreatedEvent",
                          ts,
                          payload: {
                              request_id: `plan-${sessionId}`,
                              plan_markdown: "1. Inspect workspace\n2. Draft workflow\n3. Validate outputs",
                              workflow_preview: {
                                  workflow_ir: {
                                      name: "agent-task-plan",
                                      task_configs: [
                                          {
                                              task_id: "inspect",
                                              task_type: "inspect_workspace",
                                              config: {},
                                          },
                                      ],
                                      links: [],
                                      metadata: {},
                                  },
                                  python_script: "",
                                  mermaid: "",
                                  intervention_points: [],
                              },
                          },
                      }
                    : {
                          type: "SessionCompletedEvent",
                          ts,
                          payload: { summary: `Goal completed: "${body.description}"` },
                      },
            ],
            stats: {
                ...(current.stats ?? {}),
                completedAt: planMode ? null : ts,
                durationSeconds: planMode
                    ? null
                    : (new Date(ts).getTime() - new Date(startedAt).getTime()) / 1000,
            },
        });
        if (planMode) {
            mockReviews.push({
                id: `review-plan-${taskId}-plan-${sessionId}`,
                kind: "plan",
                title: `Plan: ${body.description}`,
                description: "1. Inspect workspace",
                riskLevel: "medium",
                status: "pending",
                targetRef: {
                    type: "plan",
                    id: `plan-${sessionId}`,
                    taskId,
                    sessionId,
                },
                createdAt: ts,
                resolvedAt: null,
                resolutionComment: null,
                metadata: {},
            });
        }
    }, planMode ? 1500 : 3000);
    return session;
};

export const agentHandlers = [
    // GET /api/reviews — review queue
    http.get(`${API_BASE}/reviews`, ({ request }) => {
        const url = new URL(request.url);
        const status = url.searchParams.get("status");
        const reviews = status
            ? mockReviews.filter((review) => review.status === status)
            : mockReviews;
        return HttpResponse.json({ reviews, total: reviews.length });
    }),

    // GET /api/reviews/:reviewId — review detail
    http.get(`${API_BASE}/reviews/:reviewId`, ({ params }) => {
        const review = mockReviews.find((item) => item.id === params.reviewId);
        if (!review) {
            return HttpResponse.json(
                { detail: `Review ${params.reviewId} not found` },
                { status: 404 },
            );
        }
        return HttpResponse.json(review);
    }),

    // POST /api/reviews/:reviewId/approve — approve review
    http.post(`${API_BASE}/reviews/:reviewId/approve`, async ({ params, request }) => {
        const review = mockReviews.find((item) => item.id === params.reviewId);
        if (!review) {
            return HttpResponse.json(
                { detail: `Review ${params.reviewId} not found` },
                { status: 404 },
            );
        }
        const body = (await request.json()) as { comment?: string };
        review.status = "approved";
        review.resolvedAt = new Date().toISOString();
        review.resolutionComment = body.comment ?? null;
        return HttpResponse.json({ message: "approved" });
    }),

    // POST /api/reviews/:reviewId/reject — reject review
    http.post(`${API_BASE}/reviews/:reviewId/reject`, async ({ params, request }) => {
        const review = mockReviews.find((item) => item.id === params.reviewId);
        if (!review) {
            return HttpResponse.json(
                { detail: `Review ${params.reviewId} not found` },
                { status: 404 },
            );
        }
        const body = (await request.json()) as { comment?: string };
        review.status = "rejected";
        review.resolvedAt = new Date().toISOString();
        review.resolutionComment = body.comment ?? null;
        return HttpResponse.json({ message: "rejected" });
    }),

    // GET /api/agent-tasks — product-facing wrapper over mock sessions
    http.get(`${API_BASE}/agent-tasks`, () => {
        const tasks = getAllAgentSessions().map(taskFromSession);
        return HttpResponse.json({ tasks, total: tasks.length });
    }),

    // POST /api/agent-tasks — create a user-facing task
    http.post(`${API_BASE}/agent-tasks`, async ({ request }) => {
        const body = (await request.json()) as {
            description: string;
            plan_mode?: boolean;
            skill_id?: string | null;
        };
        const session = createMockTaskSession(body);
        return HttpResponse.json(taskFromSession(session), { status: 201 });
    }),

    // GET /api/agent-tasks/:taskId — get a single task
    http.get(`${API_BASE}/agent-tasks/:taskId`, ({ params }) => {
        const session = getAgentSession(params.taskId as string);
        if (!session) {
            return HttpResponse.json(
                { detail: `Agent task ${params.taskId} not found` },
                { status: 404 },
            );
        }
        return HttpResponse.json(taskFromSession(session));
    }),

    // GET /api/agent-tasks/:taskId/events — SSE stream of task events
    http.get(`${API_BASE}/agent-tasks/:taskId/events`, ({ params }) => {
        const session = getAgentSession(params.taskId as string);
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
            start(controller) {
                if (!session) {
                    controller.enqueue(encoder.encode('data: {"type":"error","message":"Task not found"}\n\n'));
                    controller.close();
                    return;
                }
                controller.enqueue(
                    encoder.encode(
                        session.status !== "running"
                            ? 'data: {"type":"done"}\n\n'
                            : 'data: {"type":"waiting"}\n\n',
                    ),
                );
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

    // POST /api/agent-tasks/:taskId/approve — respond to approval request
    http.post(`${API_BASE}/agent-tasks/:taskId/approve`, async ({ params, request }) => {
        const session = getAgentSession(params.taskId as string);
        if (!session) {
            return HttpResponse.json(
                { detail: `Agent task ${params.taskId} not found` },
                { status: 404 },
            );
        }
        const body = (await request.json()) as { request_id: string; approved: boolean };
        return HttpResponse.json({ request_id: body.request_id, approved: body.approved, applied: true });
    }),

    // POST /api/agent-tasks/:taskId/plan-decision — resolve a plan handoff
    http.post(`${API_BASE}/agent-tasks/:taskId/plan-decision`, async ({ params, request }) => {
        const session = getAgentSession(params.taskId as string);
        if (!session) {
            return HttpResponse.json(
                { detail: `Agent task ${params.taskId} not found` },
                { status: 404 },
            );
        }
        const body = (await request.json()) as {
            request_id: string;
            approved: boolean;
            feedback?: string;
        };
        const ts = new Date().toISOString();
        setAgentSession({
            ...session,
            status: body.approved ? "completed" : session.status,
            planMode: body.approved ? false : session.planMode,
            events: [
                ...(session.events ?? []),
                {
                    type: body.approved ? "SessionCompletedEvent" : "ObservationEvent",
                    ts,
                    payload: body.approved
                        ? { summary: "Plan approved and completed." }
                        : { content: `User rejected the plan: ${body.feedback || "(no feedback)"}` },
                },
            ],
            stats: {
                ...(session.stats ?? {}),
                completedAt: body.approved ? ts : session.stats?.completedAt,
            },
        });
        return HttpResponse.json({ message: body.approved ? "approved" : "rejected" });
    }),

    // POST /api/agent-tasks/:taskId/messages — chat message from user
    http.post(`${API_BASE}/agent-tasks/:taskId/messages`, async ({ params, request }) => {
        const session = getAgentSession(params.taskId as string);
        if (!session) {
            return HttpResponse.json(
                { detail: `Agent task ${params.taskId} not found` },
                { status: 404 },
            );
        }
        const body = (await request.json()) as { content: string; request_id?: string | null };
        const ts = new Date().toISOString();
        setAgentSession({
            ...session,
            events: [
                ...(session.events ?? []),
                {
                    type: "UserMessageEvent",
                    ts,
                    payload: { content: body.content, request_id: body.request_id ?? null },
                },
            ],
        });
        return HttpResponse.json({ message: "queued" });
    }),

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
            planMode,
            skillId: body.skill_id ?? null,
        };

        setAgentSession(newSession);

        // Simulate session completing after a short delay (mock only)
        setTimeout(() => {
            const session = getAgentSession(sessionId);
            if (!session) return;
            const completedAt = new Date().toISOString();
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
            // Plan mode always emits a workflow plan: every step is a
            // node, including investigation steps. The synthetic plan
            // below mixes inspect_dataset (investigation) with concrete
            // execution steps so the demo exercises the full IR.
            const workflowPlanMarkdown = [
                "1. list_projects() — discover what projects exist in this workspace.",
                "2. list_experiments(project_id=p1) — identify the relevant ablation.",
                "3. get_run_results(...) — pull the metric series for plotting.",
                "4. run_python(code=...) — produce a Plotly scatter chart.",
            ].join("\n");
            const workflowIr = {
                name: "energy-vs-temperature",
                task_configs: [
                    { task_id: "projects", task_type: "list_projects", config: {} },
                    {
                        task_id: "experiments",
                        task_type: "list_experiments",
                        config: { project_id: "p1" },
                    },
                    { task_id: "fetch", task_type: "get_run_results", config: {} },
                    {
                        task_id: "plot",
                        task_type: "run_python",
                        config: { script: "render_plot.py" },
                    },
                ],
                links: [
                    { source: "projects", target: "experiments" },
                    { source: "experiments", target: "fetch" },
                    { source: "fetch", target: "plot" },
                ],
                metadata: {},
            };
            const pythonScript = [
                "from molexp.workflow.spec import WorkflowSpec",
                "",
                `WORKFLOW_IR = ${JSON.stringify(workflowIr, null, 4)}`,
                "",
                "spec = WorkflowSpec.from_dict(WORKFLOW_IR)",
            ].join("\n");
            const eventsTail = planMode
                ? [
                      {
                          type: "PlanCreatedEvent",
                          ts: completedAt,
                          payload: {
                              request_id: `plan-${sessionId}`,
                              plan_markdown: workflowPlanMarkdown,
                              workflow_preview: {
                                  workflow_ir: workflowIr,
                                  python_script: pythonScript,
                                  mermaid: "",
                                  intervention_points: [
                                      "swap list_projects for list_experiments to scope tighter",
                                      "rename 'plot' to its real chart kind",
                                  ],
                              },
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
            // Plan-mode sessions stay "running" while waiting on the user's
            // plan-decision; they only flip to "completed" after approval +
            // execution finishes. Non-plan sessions complete normally.
            const tailedSession: ApiAgentSession = {
                ...session,
                status: planMode ? "running" : "completed",
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
                    completedAt: planMode ? null : completedAt,
                    durationSeconds: planMode
                        ? null
                        : (new Date(completedAt).getTime() - new Date(startedAt).getTime()) / 1000,
                },
            };
            setAgentSession(tailedSession);
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
    //
    // The mock has no real pub-sub: existing events are delivered via the
    // initial GET /sessions/:id, and writes by other handlers (plan-decision,
    // messages) update the session record but cannot push out-of-band. So
    // the SSE handler ONLY emits a control sentinel and closes — the client
    // reads new state via polling. Replaying historical events here causes
    // the UI to duplicate them on every reconnect.
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

    // POST /api/agent/sessions/:sessionId/plan-decision — resolve a plan handoff
    http.post(
        `${API_BASE}/agent/sessions/:sessionId/plan-decision`,
        async ({ params, request }) => {
            const session = getAgentSession(params.sessionId as string);
            if (!session) {
                return HttpResponse.json(
                    { detail: `Session ${params.sessionId} not found` },
                    { status: 404 },
                );
            }
            const body = (await request.json()) as {
                request_id: string;
                approved: boolean;
                edited_plan?: string | null;
                edited_workflow_ir?: Record<string, unknown> | null;
                feedback?: string;
            };
            const decisionTs = new Date().toISOString();
            // Mock semantics:
            // - approval: flip session to "completed" with a
            //   SessionCompletedEvent (the agent has nothing left to do).
            // - rejection: stay running, append an observation echoing
            //   the feedback so the agent can revise.
            if (body.approved) {
                const startedAt = session.stats?.startedAt ?? decisionTs;
                const updated: ApiAgentSession = {
                    ...session,
                    status: "completed",
                    planMode: false,
                    events: [
                        ...(session.events ?? []),
                        {
                            type: "SessionCompletedEvent",
                            ts: decisionTs,
                            payload: {
                                summary:
                                    "Workflow plan approved. Executed the steps end-to-end. " +
                                    "(mock — wire the live agent to see real output.)",
                                produced_runs: [],
                            },
                        },
                    ],
                    stats: {
                        ...(session.stats ?? {
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
                        }),
                        completedAt: decisionTs,
                        durationSeconds:
                            (new Date(decisionTs).getTime() - new Date(startedAt).getTime()) /
                            1000,
                    },
                };
                setAgentSession(updated);
            } else {
                const updated: ApiAgentSession = {
                    ...session,
                    events: [
                        ...(session.events ?? []),
                        {
                            type: "ObservationEvent",
                            ts: decisionTs,
                            payload: {
                                content:
                                    "User rejected the plan: " +
                                    (body.feedback || "(no feedback)"),
                            },
                        },
                    ],
                };
                setAgentSession(updated);
            }
            return HttpResponse.json({
                message: body.approved ? "approved" : "rejected",
            });
        },
    ),

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
