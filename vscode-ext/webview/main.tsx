/**
 * Webview entry — mounts the reused molexp `<WorkflowPreview>` and feeds it the
 * file text pushed from the extension host.
 *
 * The component is imported straight from the molexp UI's shadcn-style workflow
 * module (`@/components/workflow`, aliased to `molexp/ui/src` at build time), so
 * there is no duplicated renderer: the extension and `molexp serve` share one.
 */

import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { WorkflowPreview } from "@/components/workflow";
// Styling is delivered as two prebuilt stylesheets linked by the host HTML:
//   - theme.css   (Tailwind v4 + shadcn tokens, compiled from webview/index.css)
//   - webview.css (flowgram canvas CSS, emitted by esbuild from the JS graph)

interface VsCodeApi {
  postMessage(msg: unknown): void;
}
declare function acquireVsCodeApi(): VsCodeApi;

const vscode = acquireVsCodeApi();

const App = (): JSX.Element => {
  const [source, setSource] = useState<string | null>(null);

  useEffect(() => {
    // VSCode tags <body> with vscode-dark / vscode-high-contrast; reuse the UI's
    // `.dark` token palette so the canvas matches the editor theme.
    const syncTheme = (): void => {
      const dark = document.body.classList.contains("vscode-dark") ||
        document.body.classList.contains("vscode-high-contrast");
      document.documentElement.classList.toggle("dark", dark);
    };
    syncTheme();
    const observer = new MutationObserver(syncTheme);
    observer.observe(document.body, { attributes: true, attributeFilter: ["class"] });

    const onMessage = (event: MessageEvent): void => {
      const msg = event.data as { type?: string; text?: string };
      if (msg?.type === "update") setSource(msg.text ?? "");
    };
    window.addEventListener("message", onMessage);
    // Tell the host we are ready for the initial payload.
    vscode.postMessage({ type: "ready" });
    return () => {
      window.removeEventListener("message", onMessage);
      observer.disconnect();
    };
  }, []);

  if (source === null) {
    return <div className="p-4 text-sm text-muted-foreground">Loading workflow…</div>;
  }

  return (
    <div className="h-screen w-screen bg-background p-3 text-foreground">
      <WorkflowPreview source={source} height={window.innerHeight - 24} />
    </div>
  );
};

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
