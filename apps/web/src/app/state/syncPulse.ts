/**
 * Lightweight pub/sub for "workspace sync completed" — drives the bottom
 * status-strip heartbeat so every poll / manual refresh breathes once.
 */

import { useSyncExternalStore } from "react";

let generation = 0;
const subscribers = new Set<() => void>();

const notify = (): void => {
  for (const fn of subscribers) fn();
};

/** Bump the pulse generation — call when a poll or refresh finishes. */
export const pulseSync = (): void => {
  generation += 1;
  notify();
};

export const getSyncPulseGeneration = (): number => generation;

const subscribe = (fn: () => void): (() => void) => {
  subscribers.add(fn);
  return () => {
    subscribers.delete(fn);
  };
};

/** React hook: re-renders whenever `pulseSync()` runs. */
export const useSyncPulse = (): number =>
  useSyncExternalStore(subscribe, getSyncPulseGeneration, getSyncPulseGeneration);
