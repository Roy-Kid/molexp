/**
 * Mock handlers for the Remote Operations (molq plugin) endpoints.
 */

import { http, HttpResponse } from "msw";

const API_BASE = "/api/plugins/molq";

const now = () => new Date().toISOString();

const targets = [
  {
    name: "dardel",
    scheduler: "slurm",
    clusterName: "dardel.scilifelab.se",
    jobsDir: "/scratch/jdoe/.molq/jobs",
    healthy: true,
    healthReason: null,
    activeJobs: 3,
  },
  {
    name: "alvis",
    scheduler: "slurm",
    clusterName: "alvis.scilifelab.se",
    jobsDir: null,
    healthy: true,
    healthReason: null,
    activeJobs: 1,
  },
  {
    name: "local",
    scheduler: "local",
    clusterName: "this-machine",
    jobsDir: null,
    healthy: true,
    healthReason: null,
    activeJobs: 0,
  },
  {
    name: "aws-gpu",
    scheduler: "slurm",
    clusterName: "aws.us-east-1",
    jobsDir: null,
    healthy: false,
    healthReason: "Connection refused",
    activeJobs: 0,
  },
];

const baseSubmitted = Date.now() - 4 * 3600 * 1000;

const buildJobs = () => {
  const states = ["running", "running", "queued", "succeeded", "succeeded", "failed", "running"];
  return states.map((state, idx) => {
    const submittedAt = new Date(baseSubmitted + idx * 5 * 60_000);
    const startedAt =
      state === "queued" ? null : new Date(submittedAt.getTime() + (60 + idx * 30) * 1000);
    const finishedAt =
      state === "succeeded" || state === "failed"
        ? new Date(submittedAt.getTime() + (3600 + idx * 60) * 1000)
        : null;
    const duration = startedAt
      ? ((finishedAt ?? new Date()).getTime() - startedAt.getTime()) / 1000
      : null;
    const target = idx % 2 === 0 ? "dardel" : "alvis";
    return {
      target,
      jobId: `job-${idx + 1}`,
      schedulerJobId: `${48290 + idx}`,
      clusterName: targets.find((t) => t.name === target)?.clusterName ?? null,
      scheduler: "slurm",
      name: ["train-allegro-lr-sweep", "nemd-conductivity-run", "gromacs-benchmark", "alphafold-batch-12", "ligand-docking-array", "md-ensemble-512", "data-prep-pipeline"][idx],
      state,
      submittedAt: submittedAt.toISOString(),
      startedAt: startedAt?.toISOString() ?? null,
      finishedAt: finishedAt?.toISOString() ?? null,
      exitCode: state === "succeeded" ? 0 : state === "failed" ? 1 : null,
      durationSeconds: duration,
      cwd: "/scratch/jdoe/runs",
    };
  });
};

const computeStats = (jobs: ReturnType<typeof buildJobs>) => {
  const running = jobs.filter((j) => j.state === "running").length;
  const pending = jobs.filter((j) => ["queued", "submitted", "created"].includes(j.state)).length;
  const failed = jobs.filter((j) =>
    ["failed", "timed_out", "cancelled", "lost"].includes(j.state),
  ).length;
  const succeeded = jobs.filter((j) => j.state === "succeeded").length;
  const waits = jobs
    .filter((j) => j.startedAt && j.submittedAt)
    .map(
      (j) => (new Date(j.startedAt as string).getTime() - new Date(j.submittedAt as string).getTime()) / 1000,
    );
  const avgWaitSeconds = waits.length
    ? waits.reduce((a, b) => a + b, 0) / waits.length
    : null;
  return { running, pending, failed, succeeded, avgWaitSeconds };
};

export const molqHandlers = [
  http.get(`${API_BASE}/targets`, () => {
    return HttpResponse.json({ targets, total: targets.length });
  }),

  http.get(`${API_BASE}/jobs`, ({ request }) => {
    const url = new URL(request.url);
    const target = url.searchParams.get("target");
    const all = buildJobs();
    const filtered = target ? all.filter((j) => j.target === target) : all;
    return HttpResponse.json({
      jobs: filtered,
      stats: computeStats(filtered),
      total: filtered.length,
    });
  }),

  http.get(`${API_BASE}/jobs/:jobId`, ({ params, request }) => {
    const url = new URL(request.url);
    const target = url.searchParams.get("target");
    const job = buildJobs().find(
      (j) => j.jobId === (params.jobId as string) && (!target || j.target === target),
    );
    if (!job) {
      return HttpResponse.json({ detail: "Job not found" }, { status: 404 });
    }
    return HttpResponse.json({
      ...job,
      failureReason: job.state === "failed" ? "Exit code 1" : null,
      metadata: { run_id: `run-${job.jobId}`, project: "demo" },
      commandDisplay: `python -m molexp.cli execute /scratch/${job.jobId}`,
      transitions: [
        {
          timestamp: job.submittedAt ?? now(),
          fromState: null,
          toState: "created",
          reason: "job created",
        },
        {
          timestamp: job.submittedAt ?? now(),
          fromState: "created",
          toState: "submitted",
          reason: null,
        },
        ...(job.startedAt
          ? [
              {
                timestamp: job.startedAt,
                fromState: "submitted",
                toState: "running",
                reason: null,
              },
            ]
          : []),
        ...(job.finishedAt
          ? [
              {
                timestamp: job.finishedAt,
                fromState: "running",
                toState: job.state,
                reason: null,
              },
            ]
          : []),
      ],
      upstreamTotal: 0,
      upstreamSatisfied: 0,
      downstreamTotal: 0,
    });
  }),

  http.get(`${API_BASE}/jobs/:jobId/logs`, ({ params }) => {
    const encoder = new TextEncoder();
    const lines = [
      `[INFO] Step 8200/10000 - loss: 0.2314 - lr: 2.1e-5`,
      `[INFO] Validation loss: 0.2167`,
      `[INFO] Checkpoint saved to /ckpts/step_8200.pt`,
      `[INFO] Step 8300/10000 - loss: 0.2281 - lr: 2.1e-5`,
      `[INFO] Validation loss: 0.2142`,
      `[INFO] (mock job ${params.jobId as string} stream end)`,
      "[stream closed]",
    ];
    const stream = new ReadableStream({
      start(controller) {
        let i = 0;
        const send = () => {
          if (i >= lines.length) {
            controller.close();
            return;
          }
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ line: lines[i] })}\n\n`),
          );
          i += 1;
          setTimeout(send, 200);
        };
        send();
      },
    });
    return new HttpResponse(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      },
    });
  }),
];
