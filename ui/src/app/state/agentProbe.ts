/**
 * Agent probe gate — module-level memoization for the lightweight
 * "is the agent stack configured?" GET probes:
 *
 *   /api/agent/health · /api/agent/provider · /api/agent/commands
 *
 * Several components fire these probes on mount (health banner, provider
 * badge, slash-command palette). When the agent provider is not configured
 * the server answers 503 for each of them, and because the browser logs
 * every failed network request, each page visit used to leave a wall of red
 * console errors. Catching the rejection cannot silence those logs — the
 * only fix is to stop issuing the requests.
 *
 * After the first 503 the outcome is cached here and subsequent calls
 * reject immediately (same error instance) without touching the network,
 * until `resetAgentProbes()` runs (e.g. after provider settings are saved)
 * or the page reloads. Successful probes are never cached — fresh data
 * still flows on every mount — and concurrent callers share one in-flight
 * request, so a burst of simultaneous mounts costs a single round trip.
 */

/** Rejection produced when a probe endpoint answers 503 (provider unconfigured). */
export class AgentUnavailableError extends Error {
  readonly status = 503;

  constructor(path: string) {
    super(
      `Agent backend unavailable (HTTP 503 from ${path}) — the agent provider is not configured.`,
    );
    this.name = "AgentUnavailableError";
  }
}

interface ProbeState<T> {
  readonly unavailable: AgentUnavailableError | null;
  readonly inflight: Promise<T> | null;
}

const probeStates = new Map<string, ProbeState<unknown>>();

/**
 * Run `prober` at most once concurrently per `key`, and never again after it
 * rejects with :class:`AgentUnavailableError` (until `resetAgentProbes`).
 * Any other outcome (success, transient network error) leaves the next call
 * free to probe again.
 */
export const probeOnce = <T>(key: string, prober: () => Promise<T>): Promise<T> => {
  const state = probeStates.get(key) as ProbeState<T> | undefined;
  if (state?.unavailable) return Promise.reject(state.unavailable);
  if (state?.inflight) return state.inflight;

  const inflight = prober().then(
    (value) => {
      probeStates.delete(key);
      return value;
    },
    (error: unknown) => {
      if (error instanceof AgentUnavailableError) {
        probeStates.set(key, { unavailable: error, inflight: null });
      } else {
        probeStates.delete(key);
      }
      throw error;
    },
  );
  probeStates.set(key, { unavailable: null, inflight });
  return inflight;
};

/**
 * Forget every cached probe outcome so the next call hits the network again.
 * Called after the provider settings are successfully saved — the one event
 * that can turn an unconfigured agent stack into a configured one without a
 * page reload.
 */
export const resetAgentProbes = (): void => {
  probeStates.clear();
};
