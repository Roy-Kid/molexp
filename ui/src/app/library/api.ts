// Raw-fetch client for the /api/library routes. Kept separate from the
// generated OpenAPI client (these routes use simple query-param scoping).

import type {
  AddNotePayload,
  LibraryIndex,
  LibraryReference,
  LibraryScope,
  LibrarySource,
} from "./types";

const scopeParams = (scope: LibraryScope): string => {
  const params = new URLSearchParams();
  if (scope.projectId) params.set("project_id", scope.projectId);
  if (scope.experimentId) params.set("experiment_id", scope.experimentId);
  if (scope.runId) params.set("run_id", scope.runId);
  const qs = params.toString();
  return qs ? `?${qs}` : "";
};

const ensureOk = (response: Response): Response => {
  if (!response.ok) {
    throw new Error(`Library request failed: ${response.status} ${response.statusText}`);
  }
  return response;
};

export const libraryApi = {
  getIndex: async (scope: LibraryScope = {}): Promise<LibraryIndex> => {
    const response = ensureOk(await fetch(`/api/library${scopeParams(scope)}`));
    return response.json();
  },

  noteContent: async (assetId: string): Promise<string> => {
    const response = ensureOk(await fetch(`/api/assets/${encodeURIComponent(assetId)}/content`));
    return response.text();
  },

  updateNote: async (scope: LibraryScope, assetId: string, content: string): Promise<void> => {
    ensureOk(
      await fetch(`/api/library/notes/${encodeURIComponent(assetId)}${scopeParams(scope)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      }),
    );
  },

  addNote: async (scope: LibraryScope, payload: AddNotePayload): Promise<void> => {
    ensureOk(
      await fetch(`/api/library/notes${scopeParams(scope)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
  },

  addReference: async (
    scope: LibraryScope,
    reference: Partial<LibraryReference> & { key: string; title: string },
  ): Promise<void> => {
    ensureOk(
      await fetch(`/api/library/references${scopeParams(scope)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reference),
      }),
    );
  },

  deleteReference: async (scope: LibraryScope, key: string): Promise<void> => {
    ensureOk(
      await fetch(`/api/library/references/${encodeURIComponent(key)}${scopeParams(scope)}`, {
        method: "DELETE",
      }),
    );
  },

  importZotero: async (scope: LibraryScope, path: string): Promise<{ imported: number }> => {
    const response = ensureOk(
      await fetch(`/api/library/zotero/import${scopeParams(scope)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      }),
    );
    return response.json();
  },

  getSources: async (scope: LibraryScope = {}): Promise<LibrarySource[]> => {
    const response = ensureOk(await fetch(`/api/library/sources${scopeParams(scope)}`));
    return response.json();
  },
};
