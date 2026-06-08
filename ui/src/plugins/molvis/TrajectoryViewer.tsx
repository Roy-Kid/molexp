import { useEffect, useRef, useState } from "react";
import { workspaceApi } from "@/app/state/api";
import { cn } from "@/lib/utils";
import type { DiscoveredFile } from "@/plugins/types";

interface TrajectoryViewerProps {
  projectId: string;
  experimentId: string;
  runId: string;
  file: DiscoveredFile;
  /** Outer container classes. Defaults to a fixed-height card; pass ``h-full``
   *  to let the canvas fill a flex parent. */
  className?: string;
}

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

export const TrajectoryViewer = ({
  projectId,
  experimentId,
  runId,
  file,
  className,
}: TrajectoryViewerProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<MolvisHandle | null>(null);
  const [state, setState] = useState<LoadState>({ kind: "idle" });

  useEffect(() => {
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

        const response = await workspaceApi.getRunFileText(
          projectId,
          experimentId,
          runId,
          file.relPath,
        );
        if (cancelled) return;

        await loadFileContent(
          app as unknown as Parameters<typeof loadFileContent>[0],
          response.content,
          file.name,
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
  }, [projectId, experimentId, runId, file.relPath, file.name]);

  return (
    // ``isolate`` establishes a new stacking context so molvis-core's injected
    // overlays (``.molvis-ui-overlay`` z-index:1000, context menu z-index:10000)
    // stay scoped to the canvas and never paint above the molexp UI chrome.
    <div
      className={cn(
        "relative w-full overflow-hidden rounded-md border border-border bg-black/5 isolate",
        className ?? "h-[420px]",
      )}
    >
      <div ref={containerRef} className="absolute inset-0" />
      {state.kind === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
          Loading {file.name}…
        </div>
      )}
      {state.kind === "error" && (
        <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-xs text-destructive">
          Failed to render trajectory: {state.message}
        </div>
      )}
    </div>
  );
};
