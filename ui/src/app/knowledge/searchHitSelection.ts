/**
 * Route a knowledge-search hit to the right surface.
 *
 * `Bundle.search` walks EVERY OKF concept, so hits include workspace entities
 * (runs / experiments / projects) alongside notes. An entity hit must open its
 * entity page — sending it to `/knowledge/<identity-path>` lands on a dead-end
 * browse view with no run state on it.
 */

import type { Selection } from "@/app/types";

const RUN_PATH = /^projects\/([^/]+)\/experiments\/([^/]+)\/runs\/run-([^/]+)$/;
const EXPERIMENT_PATH = /^projects\/([^/]+)\/experiments\/([^/]+)$/;
const PROJECT_PATH = /^projects\/([^/]+)$/;

/** The Selection a search hit should open — entity kinds go to entity pages. */
export const selectionForSearchHit = (hit: { path: string; type: string }): Selection => {
  if (hit.type === "workspace.run") {
    const match = hit.path.match(RUN_PATH);
    if (match) return { objectType: "run", objectId: match[3] };
  }
  if (hit.type === "workspace.experiment") {
    const match = hit.path.match(EXPERIMENT_PATH);
    if (match) return { objectType: "experiment", objectId: match[2] };
  }
  if (hit.type === "workspace.project") {
    const match = hit.path.match(PROJECT_PATH);
    if (match) return { objectType: "project", objectId: match[1] };
  }
  // Notes, references, knowledge items — and any entity whose identity path
  // does not parse — stay on the knowledge surface.
  return { objectType: "knowledge", objectId: hit.path };
};
