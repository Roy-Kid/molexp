import { useEffect, useMemo, useState } from "react";
import type { RunFilesResponse } from "@/app/state/api";
import { workspaceApi } from "@/app/state/api";
import type { SemanticObjectType } from "@/app/types";
import type { DiscoveredPlugin } from "@/lib/file-type-discovery";
import { discoverPluginsForObject, flattenFileNodes } from "@/lib/file-type-discovery";

interface RunCoords {
  projectId: string;
  experimentId: string;
  runId: string;
}

interface UseDiscoveredFileTypesResult {
  discovered: DiscoveredPlugin[];
  loading: boolean;
  error: string | null;
}

export const useDiscoveredFileTypesForRun = (
  coords: RunCoords | null,
  objectType: SemanticObjectType = "run",
): UseDiscoveredFileTypesResult => {
  const [response, setResponse] = useState<RunFilesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!coords) {
      setResponse(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    workspaceApi
      .getRunFiles(coords.projectId, coords.experimentId, coords.runId)
      .then((value) => {
        if (!cancelled) {
          setResponse(value);
        }
      })
      .catch((reason) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "Failed to load run files");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [coords]);

  const discovered = useMemo(() => {
    if (!response) {
      return [];
    }
    const files = flattenFileNodes(response.nodes);
    return discoverPluginsForObject(objectType, files);
  }, [response, objectType]);

  return { discovered, loading, error };
};
