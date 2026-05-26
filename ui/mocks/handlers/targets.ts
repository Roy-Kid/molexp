/**
 * MSW mock handlers for the workspace ComputeTarget registry.
 *
 * The dev:mock store is in-memory — refreshing the page resets it.
 */

import { http, HttpResponse } from "msw";
import type { TargetCreateRequest } from "@/api/generated/models/TargetCreateRequest";
import { TargetResponse } from "@/api/generated/models/TargetResponse";

const API_BASE = "/api";

const targets = new Map<string, TargetResponse>();

const seed = (t: Omit<TargetResponse, "isRemote" | "defaultResources" | "defaultScheduling">) => {
  targets.set(t.name, {
    ...t,
    isRemote: t.host != null,
    defaultResources: {},
    defaultScheduling: {},
  });
};

seed({
  name: "laptop",
  scratchRoot: "/tmp/molexp",
  scheduler: TargetResponse.scheduler.LOCAL,
  host: null,
  port: null,
  identityFile: null,
  sshOpts: [],
});
seed({
  name: "hpc-slurm",
  scratchRoot: "/scratch/me/molexp",
  scheduler: TargetResponse.scheduler.SLURM,
  host: "me@hpc.example.org",
  port: null,
  identityFile: null,
  sshOpts: [],
});

const toResponse = (t: TargetResponse) => t;

export const targetsHandlers = [
  http.get(`${API_BASE}/targets`, () => {
    const list = Array.from(targets.values()).map(toResponse);
    return HttpResponse.json({ targets: list, total: list.length });
  }),

  http.post(`${API_BASE}/targets`, async ({ request }) => {
    const body = (await request.json()) as TargetCreateRequest;
    if (targets.has(body.name)) {
      return HttpResponse.json(
        { detail: `compute target '${body.name}' already exists` },
        { status: 409 },
      );
    }
    const created: TargetResponse = {
      name: body.name,
      scratchRoot: body.scratchRoot,
      scheduler: (body.scheduler ?? TargetResponse.scheduler.LOCAL) as TargetResponse.scheduler,
      host: body.host ?? null,
      port: body.port ?? null,
      identityFile: body.identityFile ?? null,
      sshOpts: body.sshOpts ?? [],
      isRemote: !!body.host,
      defaultResources: {},
      defaultScheduling: {},
    };
    targets.set(created.name, created);
    return HttpResponse.json(created, { status: 201 });
  }),

  http.delete(`${API_BASE}/targets/:name`, ({ params }) => {
    const name = String(params.name);
    if (!targets.has(name)) {
      return HttpResponse.json(
        { detail: `no compute target named '${name}'` },
        { status: 404 },
      );
    }
    targets.delete(name);
    return new HttpResponse(null, { status: 204 });
  }),

  http.post(`${API_BASE}/targets/:name/test`, ({ params }) => {
    const name = String(params.name);
    const target = targets.get(name);
    if (!target) {
      return HttpResponse.json(
        { detail: `no compute target named '${name}'` },
        { status: 404 },
      );
    }
    if (target.isRemote) {
      return HttpResponse.json({
        name,
        ok: false,
        error: "mock environment cannot reach remote hosts",
        checks: [
          { label: "command execution", ok: false, detail: "ssh disabled in mock" },
        ],
      });
    }
    return HttpResponse.json({
      name,
      ok: true,
      error: null,
      checks: [
        { label: "command execution", ok: true, detail: null },
        { label: `mkdir ${target.scratchRoot}`, ok: true, detail: null },
        { label: "file round-trip", ok: true, detail: null },
      ],
    });
  }),
];
