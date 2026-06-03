// Flattens the workspace snapshot into a single searchable list of EntityRefs.
// This is the backing index for the global command palette and any
// "jump to anything" surface — one list, every kind, ranked uniformly.

import type { EntityRef } from "@/app/entities/kinds";
import type { WorkspaceSnapshot } from "@/app/types";

export interface CatalogEntry {
  ref: EntityRef;
  /** Lower-cased haystack: name + id + summary, for substring matching. */
  haystack: string;
}

const entry = (ref: EntityRef, ...extra: (string | null | undefined)[]): CatalogEntry => ({
  ref,
  haystack: [ref.label, ref.id, ...extra].filter(Boolean).join(" ").toLowerCase(),
});

export const buildCatalog = (snapshot: WorkspaceSnapshot): CatalogEntry[] => {
  const entries: CatalogEntry[] = [];

  for (const p of snapshot.projects) {
    entries.push(entry({ kind: "project", id: p.id, label: p.name, status: p.status }, p.summary));
  }
  for (const e of snapshot.experiments) {
    entries.push(
      entry({ kind: "experiment", id: e.id, label: e.name, status: e.status }, e.summary),
    );
  }
  for (const r of snapshot.runs) {
    entries.push(
      entry({ kind: "run", id: r.id, label: r.name || r.id, status: r.status }, r.summary),
    );
  }
  for (const w of snapshot.workflows) {
    entries.push(entry({ kind: "workflow", id: w.id, label: w.name, status: w.status }, w.summary));
  }
  for (const a of snapshot.assets) {
    entries.push(entry({ kind: "asset", id: a.id, label: a.name, status: a.status }, a.summary));
  }
  for (const s of snapshot.agentSessions) {
    entries.push(entry({ kind: "agent", id: s.id, label: s.goal, status: s.status }));
  }

  return entries;
};

/** Substring-rank a catalog against a query. Exact-prefix matches first, then
 *  word-boundary, then any substring. Empty query returns the head of the list. */
export const searchCatalog = (
  catalog: CatalogEntry[],
  query: string,
  limit = 30,
): CatalogEntry[] => {
  const q = query.trim().toLowerCase();
  if (!q) return catalog.slice(0, limit);

  const scored: { entry: CatalogEntry; score: number }[] = [];
  for (const e of catalog) {
    const idx = e.haystack.indexOf(q);
    if (idx < 0) continue;
    // Lower score = better. Prefix beats word-boundary beats mid-string.
    const score = idx === 0 ? 0 : e.haystack[idx - 1] === " " ? 1 : 2;
    scored.push({ entry: e, score: score * 1000 + idx });
  }
  scored.sort((a, b) => a.score - b.score);
  return scored.slice(0, limit).map((s) => s.entry);
};
