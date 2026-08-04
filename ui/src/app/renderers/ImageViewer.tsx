import { useEffect, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import type { RendererProps } from "@/app/types";
import { WorkbenchOperationState, WorkbenchRetryAction } from "@/components/workbench";

export const ImageViewer = ({ selection }: RendererProps): JSX.Element => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    void tick;
    if (selection.objectType !== "workspace-file") {
      return;
    }

    let revoked = false;
    let currentUrl: string | null = null;
    setLoading(true);
    setError(null);
    setImageUrl(null);

    workspaceApi
      .getWorkspaceFileBlob(selection.objectId)
      .then((blob) => {
        if (revoked) {
          return;
        }
        currentUrl = URL.createObjectURL(blob);
        setImageUrl(currentUrl);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to load image");
        setImageUrl(null);
      })
      .finally(() => {
        if (!revoked) setLoading(false);
      });

    return () => {
      revoked = true;
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl);
      }
    };
  }, [selection, tick]);

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <header className="flex h-10 flex-none items-center border-b border-border px-3">
        <p className="min-w-0 truncate font-mono text-label text-muted-foreground tabular-nums">
          {selection.objectId}
        </p>
      </header>
      <div className="flex min-h-0 flex-1 items-center justify-center p-3">
        {loading && !imageUrl && <WorkbenchOperationState kind="loading" title="Loading image…" />}
        {error && (
          <WorkbenchOperationState
            kind="error"
            title="Could not load image"
            detail={error}
            action={<WorkbenchRetryAction onClick={() => setTick((t) => t + 1)} />}
          />
        )}
        {!error && imageUrl && (
          <img
            src={imageUrl}
            alt={selection.objectId}
            className="mol-motion-enter-fade max-h-full max-w-full object-contain"
          />
        )}
      </div>
    </div>
  );
};
