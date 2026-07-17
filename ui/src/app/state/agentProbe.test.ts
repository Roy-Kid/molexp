/**
 * Tests for the agent probe gate (agentProbe.ts) and its wiring into the
 * three probe endpoints in api.ts.
 *
 * Runs in rstest's node environment (no jsdom): probeOnce is exercised
 * directly as a pure async function, and the api.ts wiring is verified
 * behaviorally by stubbing `globalThis.fetch` — a 503 must be requested
 * exactly once, with every later call rejecting from the module cache
 * without touching the network.
 */

import { afterEach, beforeEach, describe, expect, it } from "@rstest/core";
import { AgentUnavailableError, probeOnce, resetAgentProbes } from "@/app/state/agentProbe";
import { agentAdminApi, agentApi } from "@/app/state/api";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  resetAgentProbes();
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  resetAgentProbes();
});

describe("probeOnce", () => {
  it("caches an AgentUnavailableError and never re-invokes the prober", async () => {
    let calls = 0;
    const prober = async (): Promise<string> => {
      calls += 1;
      throw new AgentUnavailableError("/api/agent/health");
    };
    await expect(probeOnce("k", prober)).rejects.toBeInstanceOf(AgentUnavailableError);
    await expect(probeOnce("k", prober)).rejects.toBeInstanceOf(AgentUnavailableError);
    await expect(probeOnce("k", prober)).rejects.toBeInstanceOf(AgentUnavailableError);
    expect(calls).toBe(1);
  });

  it("does not cache success — every call probes again", async () => {
    let calls = 0;
    const prober = async (): Promise<number> => {
      calls += 1;
      return calls;
    };
    expect(await probeOnce("k", prober)).toBe(1);
    expect(await probeOnce("k", prober)).toBe(2);
    expect(calls).toBe(2);
  });

  it("does not cache other errors — the next call retries", async () => {
    let calls = 0;
    const prober = async (): Promise<never> => {
      calls += 1;
      throw new Error("network blip");
    };
    await expect(probeOnce("k", prober)).rejects.toThrow("network blip");
    await expect(probeOnce("k", prober)).rejects.toThrow("network blip");
    expect(calls).toBe(2);
  });

  it("shares one in-flight request between concurrent callers", async () => {
    let calls = 0;
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const prober = async (): Promise<string> => {
      calls += 1;
      await gate;
      return "ok";
    };
    const first = probeOnce("k", prober);
    const second = probeOnce("k", prober);
    release?.();
    expect(await first).toBe("ok");
    expect(await second).toBe("ok");
    expect(calls).toBe(1);
  });

  it("keys the cache per probe", async () => {
    await expect(
      probeOnce("a", async () => {
        throw new AgentUnavailableError("/a");
      }),
    ).rejects.toBeInstanceOf(AgentUnavailableError);
    expect(await probeOnce("b", async () => "alive")).toBe("alive");
  });

  it("resetAgentProbes clears a cached unavailable outcome", async () => {
    let calls = 0;
    const prober = async (): Promise<never> => {
      calls += 1;
      throw new AgentUnavailableError("/api/agent/provider");
    };
    await expect(probeOnce("k", prober)).rejects.toBeInstanceOf(AgentUnavailableError);
    resetAgentProbes();
    await expect(probeOnce("k", prober)).rejects.toBeInstanceOf(AgentUnavailableError);
    expect(calls).toBe(2);
  });
});

describe("api.ts probe wiring (503 requested at most once per endpoint)", () => {
  const install503Fetch = (): { calls: string[] } => {
    const calls: string[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      calls.push(String(input));
      return new Response(JSON.stringify({ detail: "agent routes disabled" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      });
    }) as typeof fetch;
    return { calls };
  };

  it("agentApi.getHealth stops probing after the first 503", async () => {
    const { calls } = install503Fetch();
    await expect(agentApi.getHealth()).rejects.toBeInstanceOf(AgentUnavailableError);
    await expect(agentApi.getHealth()).rejects.toBeInstanceOf(AgentUnavailableError);
    expect(calls).toEqual(["/api/agent/health"]);
  });

  it("empty skill and tool catalogs tolerate a legacy server 503", async () => {
    const { calls } = install503Fetch();
    await expect(agentAdminApi.listSkills()).resolves.toEqual([]);
    await expect(agentAdminApi.listSkills()).resolves.toEqual([]);
    await expect(agentAdminApi.listToolsAndGroups()).resolves.toEqual({ tools: [], mcpGroups: [] });
    await expect(agentAdminApi.listToolsAndGroups()).resolves.toEqual({ tools: [], mcpGroups: [] });
    expect(calls).toEqual(["/api/agent/skills", "/api/agent/tools"]);
  });

  it("an unavailable MCP service remains distinguishable from an empty catalog", async () => {
    const { calls } = install503Fetch();
    await expect(agentAdminApi.listMcpServers()).rejects.toBeInstanceOf(AgentUnavailableError);
    await expect(agentAdminApi.listMcpServers()).rejects.toBeInstanceOf(AgentUnavailableError);
    expect(calls).toEqual(["/api/agent/mcp/servers"]);
  });

  it("a successful updateProvider resets the probe cache", async () => {
    const { calls } = install503Fetch();
    await expect(agentApi.getHealth()).rejects.toBeInstanceOf(AgentUnavailableError);

    // Provider save succeeds → cache cleared → health probes again.
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      calls.push(String(input));
      return new Response(
        JSON.stringify({ ready: true, provider: "p", model: "m", source: "stored" }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }) as typeof fetch;
    await agentAdminApi.updateProvider({ model: "m" });
    await agentApi.getHealth();
    expect(calls).toEqual(["/api/agent/health", "/api/agent/provider", "/api/agent/health"]);
  });
});
