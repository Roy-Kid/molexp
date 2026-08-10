/**
 * Inline molvis viewer for chat embed_structure artifacts (content string).
 *
 * molvis GUIManager appends ``.molvis-ui-overlay`` with ``position: absolute``
 * onto the **mount container**. The container must be a positioned box with
 * non-zero size (see TrajectoryViewer) — otherwise panels like "Persp" attach
 * to a higher ancestor and float at the page top-left.
 */

import { type JSX, useEffect, useRef, useState } from "react";

type LoadState = "loading" | "ready" | "error";

interface MolvisHandle {
  destroy?: () => void;
  enableFitContainer?: (v: boolean) => void;
  start?: () => Promise<void> | void;
  resize?: () => void;
}

export const InlineStructureViewer = ({
  content,
  filename,
  title,
}: {
  content: string;
  filename: string;
  title?: string;
}): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<MolvisHandle | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !content) return;

    let cancelled = false;
    setState("loading");
    setError(null);

    const run = async (): Promise<void> => {
      try {
        const [{ mountMolvis }, { loadFileContent }] = await Promise.all([
          import("@molcrafts/molvis-stage"),
          import("@molcrafts/molvis-stage/io"),
        ]);
        if (cancelled) return;

        // Ensure laid-out size before molvis reads clientWidth/Height.
        await new Promise<void>((r) => requestAnimationFrame(() => r()));

        const app = mountMolvis(container) as unknown as MolvisHandle;
        appRef.current = app;
        app.enableFitContainer?.(true);
        if (typeof app.start === "function") await app.start();
        if (cancelled) return;

        await loadFileContent(
          app as unknown as Parameters<typeof loadFileContent>[0],
          content,
          filename || "structure.xyz",
        );
        if (cancelled) return;

        // Second resize after DOM settles — canvas + overlay fill the box.
        app.resize?.();
        requestAnimationFrame(() => {
          if (!cancelled) app.resize?.();
        });
        setState("ready");
      } catch (reason) {
        if (cancelled) return;
        setState("error");
        setError(reason instanceof Error ? reason.message : String(reason));
      }
    };
    void run();

    return () => {
      cancelled = true;
      const app = appRef.current;
      appRef.current = null;
      if (app && typeof app.destroy === "function") {
        try {
          app.destroy();
        } catch {
          // teardown errors are non-fatal
        }
      }
    };
  }, [content, filename]);

  return (
    <div className="overflow-hidden rounded-panel border border-border/60 bg-surface">
      {(title ?? filename) && (
        <div className="flex items-center justify-between border-b border-border/50 bg-muted/30 px-3 py-1">
          <span className="truncate font-mono text-micro text-muted-foreground">
            {title ?? filename}
          </span>
          <span className="text-micro text-muted-foreground">
            {state === "loading"
              ? "loading molvis…"
              : state === "error"
                ? "viewer error"
                : "molvis"}
          </span>
        </div>
      )}
      {state === "error" ? (
        <pre className="max-h-40 overflow-auto whitespace-pre-wrap px-3 py-2 font-mono text-micro text-destructive">
          {error}
          {"\n\n"}
          {content.slice(0, 800)}
        </pre>
      ) : (
        <div
          ref={containerRef}
          style={{ width: "100%", height: "var(--spacing-structure-preview)" }}
          className="bg-canvas"
        />
      )}
    </div>
  );
};
