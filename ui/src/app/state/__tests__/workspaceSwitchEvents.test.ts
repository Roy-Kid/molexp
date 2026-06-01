/**
 * Tests for workspaceSwitchEvents — the emit/subscribe contract that
 * `RemoteWorkspacesPanel` uses to tell long-lived SSE/EventSource
 * holders that the active workspace is about to swap.
 *
 * The test env is `node` (per rstest.config) — emit() short-circuits on
 * `typeof window === "undefined"`. We stub a minimal EventTarget-shaped
 * window on `globalThis` for the duration of the test so the
 * emit/listen path runs end-to-end.
 */

import { afterEach, beforeEach, describe, expect, it } from "@rstest/core";

import {
  emitWorkspaceSwitching,
  onWorkspaceSwitching,
  type WorkspaceSwitchingDetail,
} from "../workspaceSwitchEvents";

const originalWindow = (globalThis as { window?: unknown }).window;

beforeEach(() => {
  // jsdom would be heavier; an EventTarget is the exact surface the
  // emit/subscribe pair uses, so the stub is faithful.
  const eventTarget = new EventTarget() as Window;
  (globalThis as { window: unknown }).window = eventTarget;
});

afterEach(() => {
  if (originalWindow === undefined) {
    delete (globalThis as { window?: unknown }).window;
  } else {
    (globalThis as { window: unknown }).window = originalWindow;
  }
});

describe("workspaceSwitchEvents", () => {
  it("delivers the detail payload to subscribers", () => {
    const received: WorkspaceSwitchingDetail[] = [];
    onWorkspaceSwitching((detail) => received.push(detail));

    emitWorkspaceSwitching({ activeDescriptor: "hpc-2" });

    expect(received).toEqual([{ activeDescriptor: "hpc-2" }]);
  });

  it("supports multiple subscribers independently", () => {
    const a: string[] = [];
    const b: string[] = [];
    onWorkspaceSwitching((d) => a.push(d.activeDescriptor ?? "<null>"));
    onWorkspaceSwitching((d) => b.push(d.activeDescriptor ?? "<null>"));

    emitWorkspaceSwitching({ activeDescriptor: "lab" });
    emitWorkspaceSwitching({ activeDescriptor: null });

    expect(a).toEqual(["lab", "<null>"]);
    expect(b).toEqual(["lab", "<null>"]);
  });

  it("unsubscribe removes the listener", () => {
    const received: string[] = [];
    const unsubscribe = onWorkspaceSwitching((d) => {
      received.push(d.activeDescriptor ?? "<null>");
    });

    emitWorkspaceSwitching({ activeDescriptor: "first" });
    unsubscribe();
    emitWorkspaceSwitching({ activeDescriptor: "second" });

    // Only the pre-unsubscribe event landed.
    expect(received).toEqual(["first"]);
  });

  it("emit is a no-op when window is undefined (SSR-safe)", () => {
    // Tear down the stub to simulate an SSR / non-browser host.
    delete (globalThis as { window?: unknown }).window;

    // No throw, no observable side-effect. We can't assert "no listeners
    // fired" because subscribers also short-circuit; the contract is
    // simply "doesn't crash without a window".
    expect(() => emitWorkspaceSwitching({ activeDescriptor: "x" })).not.toThrow();
  });

  it("onWorkspaceSwitching returns a no-op unsubscriber when window is undefined", () => {
    delete (globalThis as { window?: unknown }).window;

    const unsubscribe = onWorkspaceSwitching(() => {
      throw new Error("listener must not fire without window");
    });

    expect(() => unsubscribe()).not.toThrow();
  });
});
