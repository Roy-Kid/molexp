import { useEffect, useRef, useState } from "react";
import type { FilePreviewContentProps } from "@/plugins/types";

type LoadState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready" }
  | { kind: "error"; message: string };

interface MolvisHandle {
  destroy?: () => void;
  enableFitContainer?: (enabled: boolean) => void;
  start?: () => Promise<void> | void;
}

export const MolvisDatasetPreview = ({ assetId }: FilePreviewContentProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<MolvisHandle | null>(null);
  const [state, setState] = useState<LoadState>({ kind: "idle" });

  useEffect(() => {
    if (!assetId) {
      return;
    }
    const container = containerRef.current;
    if (!container) {
      return;
    }

    let cancelled = false;
    setState({ kind: "loading" });

    const run = async (): Promise<void> => {
      try {
        const [{ mountMolvis }, { loadFileContent }] = await Promise.all([
          import("@molcrafts/molvis-core"),
          import("@molcrafts/molvis-core/io"),
        ]);
        if (cancelled) return;

        const app = mountMolvis(container) as unknown as MolvisHandle;
        appRef.current = app;
        if (typeof app.enableFitContainer === "function") {
          app.enableFitContainer(true);
        }
        if (typeof app.start === "function") {
          await app.start();
        }
        if (cancelled) return;

        const res = await fetch(`/api/assets/${encodeURIComponent(assetId)}/preview?format=frames`);
        if (!res.ok) throw new Error(`preview failed: ${res.status}`);
        const content = await res.text();
        if (cancelled) return;

        await loadFileContent(
          app as unknown as Parameters<typeof loadFileContent>[0],
          content,
          "preview.xyz",
        );
        if (cancelled) return;

        setState({ kind: "ready" });
      } catch (reason) {
        if (cancelled) return;
        const message = reason instanceof Error ? reason.message : String(reason);
        setState({ kind: "error", message });
      }
    };

    run();

    return () => {
      cancelled = true;
      const app = appRef.current;
      appRef.current = null;
      if (app && typeof app.destroy === "function") {
        try {
          app.destroy();
        } catch {
          // best-effort cleanup
        }
      }
    };
  }, [assetId]);

  if (!assetId) {
    return (
      <div className="flex h-[420px] w-full items-center justify-center px-4 text-center text-xs text-muted-foreground">
        Register this file as a dataset asset to preview it.
      </div>
    );
  }

  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-md border border-border bg-black/5">
      <div ref={containerRef} className="absolute inset-0" />
      {state.kind === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
          Loading preview…
        </div>
      )}
      {state.kind === "error" && (
        <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-destructive">
          Failed to render preview: {state.message}
        </div>
      )}
    </div>
  );
};
