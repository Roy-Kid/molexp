/**
 * Cross-feature fixtures for `npm run dev:ui` (leaf `npm run dev`).
 *
 * Domain handlers own entity CRUD. This file fills the small read models that
 * span domains (knowledge, activity, plans, approvals, cache and workspaces),
 * so every navigation destination can be exercised without a Python server.
 */

import { http, HttpResponse } from "msw";

interface ShowcaseNote {
  name: string;
  relPath: string;
  excerpt: string;
  status: string;
  tags: string[];
  body: string;
  links: string[];
}

const notes = new Map<string, ShowcaseNote>(
  [
    {
      name: "AlphaFold benchmark findings",
      relPath: "notes/alphafold-benchmark",
      excerpt: "The bf16 baseline converged fastest while preserving mean pLDDT.",
      status: "published",
      tags: ["protein-folding", "benchmark"],
      body: [
        "# AlphaFold benchmark findings",
        "",
        "The `bf16` baseline converged fastest while preserving mean pLDDT.",
        "",
        "| run | batch | pLDDT | status |",
        "| --- | ---: | ---: | --- |",
        "| run-001 | 32 | 87.4 | succeeded |",
        "| run-002 | 64 | — | running |",
        "",
        "> Linked to experiment `exp-001` and checkpoint `asset-003`.",
      ].join("\n"),
      links: ["notes/gpu-retry-playbook"],
    },
    {
      name: "GPU retry playbook",
      relPath: "notes/gpu-retry-playbook",
      excerpt: "Operational notes for recovering failed MolQ and Slurm attempts.",
      status: "draft",
      tags: ["operations", "molq", "gpu"],
      body: [
        "# GPU retry playbook",
        "",
        "1. Inspect the failed execution log.",
        "2. Reduce `batch_size` by half.",
        "3. Resume with the same config hash for provenance continuity.",
      ].join("\n"),
      links: ["notes/alphafold-benchmark"],
    },
  ].map((note) => [note.relPath, note]),
);

let approvalOpen = true;

const noteSummary = (note: ShowcaseNote) => ({
  name: note.name,
  relPath: note.relPath,
  excerpt: note.excerpt,
  status: note.status,
  tags: note.tags,
});

const planSummary = {
  runId: "plan-run-001",
  projectId: "protein-folding",
  experimentId: "exp-001",
  title: "Protein folding precision sweep",
  status: "waiting_approval",
  hasWorkflow: true,
  createdAt: "2025-01-15T09:10:00.000Z",
};

export const featureShowcaseHandlers = [
  http.get("/api/workspaces", () =>
    HttpResponse.json([
      {
        key: "feature-showcase",
        label: "Feature showcase",
        isRemote: false,
        path: "/mock-workspace",
        active: true,
        unreachable: false,
      },
    ]),
  ),

  http.get("/api/events", ({ request }) => {
    const limit = Number(new URL(request.url).searchParams.get("limit") ?? 50);
    const rows = [
      {
        id: "event-005",
        seq: 5,
        type: "run.failed",
        created_at: "2025-01-15T11:56:00.000Z",
        actor: "molq",
        refs: ["protein-folding", "exp-001", "run-003"],
        payload: { reason: "CUDA out of memory", scheduler_job_id: "421188" },
      },
      {
        id: "event-004",
        seq: 4,
        type: "asset.added",
        created_at: "2025-01-15T11:30:00.000Z",
        actor: "train",
        refs: ["run-001", "asset-003"],
        payload: { kind: "checkpoint", size: 20971520 },
      },
      {
        id: "event-003",
        seq: 3,
        type: "run.completed",
        created_at: "2025-01-15T11:00:00.000Z",
        actor: "local",
        refs: ["catalyst-search", "exp-101", "run-101"],
        payload: { hit_rate: 0.31 },
      },
      {
        id: "event-002",
        seq: 2,
        type: "run.started",
        created_at: "2025-01-15T10:28:00.000Z",
        actor: "molq",
        refs: ["protein-folding", "exp-002", "run-202"],
        payload: { target: "dardel-gpu" },
      },
      {
        id: "event-001",
        seq: 1,
        type: "knowledge.created",
        created_at: "2025-01-15T09:15:00.000Z",
        actor: "researcher",
        refs: ["notes/alphafold-benchmark"],
        payload: { title: "AlphaFold benchmark findings" },
      },
    ];
    return HttpResponse.json(rows.slice(0, Number.isFinite(limit) ? limit : 50));
  }),

  http.get("/api/knowledge", ({ request }) => {
    const url = new URL(request.url);
    const tag = url.searchParams.get("tag");
    const status = url.searchParams.get("status");
    const filtered = [...notes.values()].filter(
      (note) => (!tag || note.tags.includes(tag)) && (!status || note.status === status),
    );
    return HttpResponse.json({
      notes: filtered.map(noteSummary),
      references: [
        {
          name: "alphafold-paper",
          relPath: "references/alphafold-paper",
          title: "Highly accurate protein structure prediction with AlphaFold",
          authors: ["Jumper et al."],
          venue: "Nature",
          year: 2021,
          doi: "10.1038/s41586-021-03819-2",
          url: "https://doi.org/10.1038/s41586-021-03819-2",
        },
      ],
      total: filtered.length + 1,
    });
  }),

  http.get("/api/knowledge/note", ({ request }) => {
    const path = new URL(request.url).searchParams.get("path") ?? "";
    const note = notes.get(path);
    if (!note) return HttpResponse.json({ detail: "Note not found" }, { status: 404 });
    return HttpResponse.json({
      name: note.name,
      relPath: note.relPath,
      body: note.body,
      links: note.links,
      cards: [
        { kind: "experiment", id: "exp-001", title: "AlphaFold Baseline", status: "active" },
        { kind: "run", id: "run-001", title: "run-001", status: "succeeded" },
        { kind: "asset", id: "asset-003", title: "alphafold.pt", status: "active" },
      ],
    });
  }),

  http.get("/api/knowledge/search", ({ request }) => {
    const query = (new URL(request.url).searchParams.get("q") ?? "").toLowerCase();
    const hits = [...notes.values()]
      .filter((note) => `${note.name} ${note.body}`.toLowerCase().includes(query))
      .map((note) => ({
        path: note.relPath,
        title: note.name,
        type: "note",
        snippet: note.excerpt,
        tags: note.tags,
      }));
    return HttpResponse.json({ hits, truncated: false });
  }),

  http.get("/api/knowledge/backlinks", () =>
    HttpResponse.json({ backlinks: [...notes.values()].slice(0, 1).map(noteSummary) }),
  ),

  http.get("/api/knowledge/entity-backlinks", ({ request }) => {
    const url = new URL(request.url);
    const entity =
      url.searchParams.get("run_id") ??
      url.searchParams.get("experiment_id") ??
      url.searchParams.get("project_id") ??
      "entity";
    return HttpResponse.json({
      entity,
      backlinks: [
        {
          path: "notes/alphafold-benchmark",
          title: "AlphaFold benchmark findings",
          type: "note",
          role: "records",
        },
      ],
    });
  }),

  http.put("/api/knowledge/doc", async ({ request }) => {
    const url = new URL(request.url);
    const path = url.searchParams.get("path") ?? "";
    const note = notes.get(path);
    if (!note) return HttpResponse.json({ detail: "Note not found" }, { status: 404 });
    const body = (await request.json()) as { body?: string };
    note.body = body.body ?? note.body;
    return HttpResponse.json({
      name: note.name,
      relPath: note.relPath,
      body: note.body,
      links: note.links,
      cards: [],
    });
  }),

  http.patch("/api/knowledge/doc/meta", async ({ request }) => {
    const url = new URL(request.url);
    const path = url.searchParams.get("path") ?? "";
    const note = notes.get(path);
    if (!note) return HttpResponse.json({ detail: "Note not found" }, { status: 404 });
    const body = (await request.json()) as { tags?: string[]; status?: string };
    if (body.tags) note.tags = body.tags;
    if (body.status) note.status = body.status;
    return HttpResponse.json(noteSummary(note));
  }),

  http.post("/api/knowledge/doc/embed", async ({ request }) => {
    const body = (await request.json()) as { role?: string; target?: string };
    const srcPath = new URL(request.url).searchParams.get("path") ?? "";
    return HttpResponse.json({
      srcPath,
      target: body.target ?? "run-001",
      role: body.role ?? "references",
    });
  }),

  http.get("/api/plans", () => HttpResponse.json({ plans: [planSummary], total: 1 })),
  http.get("/api/projects/:projectId/experiments/:experimentId/plans", () =>
    HttpResponse.json({
      plans: [
        {
          runId: planSummary.runId,
          title: planSummary.title,
          status: planSummary.status,
          hasWorkflow: planSummary.hasWorkflow,
          createdAt: planSummary.createdAt,
        },
      ],
      total: 1,
    }),
  ),

  http.get("/api/approvals", () => {
    const items = approvalOpen
      ? [
          {
            taskKind: "plan",
            taskId: "plan-task-001",
            requestId: "approval-001",
            projectId: "protein-folding",
            experimentId: "exp-001",
            runId: "plan-run-001",
            intent: "Create and launch the bf16 precision sweep",
            reason: "The plan will materialize a new workflow and submit GPU jobs.",
            preview: "4 runs · learning_rate × precision · dardel-gpu",
            requestedAt: "2025-01-15T11:58:00.000Z",
            scope: "experiment",
            packId: "review-pack-001",
            metadata: { risk: "medium", estimated_gpu_hours: 12 },
          },
        ]
      : [];
    return HttpResponse.json({ items, total: items.length });
  }),

  http.post("/api/approvals/:taskKind/:taskId/decisions", ({ params }) => {
    approvalOpen = false;
    return HttpResponse.json({
      taskKind: params.taskKind,
      taskId: params.taskId,
      status: "approved",
    });
  }),

  http.post(
    "/api/projects/:projectId/experiments/:experimentId/curate-tasks",
    async ({ params, request }) => {
      const body = (await request.json()) as { request?: string; model?: string };
      return HttpResponse.json({
        taskId: "curate-showcase-001",
        projectId: params.projectId,
        experimentId: params.experimentId,
        runId: "curate-run-001",
        status: "waiting_approval",
        model: body.model ?? "gpt-5",
        requestPreview: body.request ?? "Reorganize experiment outputs",
        createdAt: new Date().toISOString(),
        capabilityId: "workspace.reorganize",
        granted: null,
      });
    },
  ),

  http.get(
    "/api/projects/:projectId/experiments/:experimentId/curate-tasks/:taskId",
    ({ params }) =>
      HttpResponse.json({
        taskId: params.taskId,
        projectId: params.projectId,
        experimentId: params.experimentId,
        runId: "curate-run-001",
        status: approvalOpen ? "waiting_approval" : "completed",
        model: "gpt-5",
        requestPreview: "Reorganize experiment outputs",
        createdAt: new Date().toISOString(),
        capabilityId: "workspace.reorganize",
        granted: !approvalOpen,
        mutationSummary: approvalOpen ? null : "Moved 3 artifacts into a curated output set.",
      }),
  ),

  http.get("/api/cache/stats", () =>
    HttpResponse.json({ entryCount: 42, storeDir: "/mock-workspace/.molexp/cache" }),
  ),
  http.delete("/api/cache", () => HttpResponse.json({ removedCount: 42 })),
];
