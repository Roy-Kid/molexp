import { useCallback, useEffect, useState } from "react";
import type { NoteSummary } from "@/api/generated/models/NoteSummary";
import { workspaceApi } from "@/app/state/api";

/**
 * Hand-written data + mutation hook for the Knowledge document tree — the
 * repo's `useState` + `useEffect` + zustand idiom (hand-rolled load + imperative
 * mutations), deliberately not a query-cache library.
 * It loads every Note through {@link workspaceApi.listKnowledge} and exposes
 * imperative create / rename / move / delete verbs plus a backlinks lookup,
 * each routed through `workspaceApi` (thin wrappers over the generated
 * `KnowledgeService`). Every successful mutation triggers a refetch so the tree
 * realigns with the server-authoritative Note set.
 */
export interface UseKnowledgeDocs {
  notes: NoteSummary[];
  loading: boolean;
  error: string | null;
  reload: () => Promise<void>;
  createDoc: (name: string, parentPath?: string | null) => Promise<void>;
  renameDoc: (path: string, name: string) => Promise<void>;
  moveDoc: (path: string, parentPath: string) => Promise<void>;
  deleteDoc: (path: string) => Promise<void>;
  getBacklinks: (path: string) => Promise<NoteSummary[]>;
}

const toMessage = (err: unknown, fallback: string): string =>
  err instanceof Error ? err.message : fallback;

export const useKnowledgeDocs = (): UseKnowledgeDocs => {
  const [notes, setNotes] = useState<NoteSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async (): Promise<void> => {
    setLoading(true);
    try {
      const response = await workspaceApi.listKnowledge();
      setNotes(response.notes);
      setError(null);
    } catch (err) {
      setError(toMessage(err, "Failed to load knowledge documents."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const createDoc = useCallback(
    async (name: string, parentPath?: string | null): Promise<void> => {
      await workspaceApi.createKnowledgeDoc(name, { parentPath: parentPath ?? null });
      await reload();
    },
    [reload],
  );

  const renameDoc = useCallback(
    async (path: string, name: string): Promise<void> => {
      await workspaceApi.renameKnowledgeDoc(path, name);
      await reload();
    },
    [reload],
  );

  const moveDoc = useCallback(
    async (path: string, parentPath: string): Promise<void> => {
      await workspaceApi.moveKnowledgeDoc(path, parentPath);
      await reload();
    },
    [reload],
  );

  const deleteDoc = useCallback(
    async (path: string): Promise<void> => {
      await workspaceApi.deleteKnowledgeDoc(path);
      await reload();
    },
    [reload],
  );

  const getBacklinks = useCallback(async (path: string): Promise<NoteSummary[]> => {
    const response = await workspaceApi.getKnowledgeBacklinks(path);
    return response.backlinks;
  }, []);

  return {
    notes,
    loading,
    error,
    reload,
    createDoc,
    renameDoc,
    moveDoc,
    deleteDoc,
    getBacklinks,
  };
};
