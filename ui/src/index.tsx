// reflect-metadata MUST be imported before any flowgram canvas module so the
// editor's inversify DI containers see the emitted decorator metadata.
import "reflect-metadata";
import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { RouteErrorBoundary } from "@/app/layout/RouteErrorBoundary";
import { bootPlugins } from "@/plugins/runtime";
import App from "./App";
import "./styles/tailwind.css";

// A data router (vs. the plain <BrowserRouter>) is required so in-app navigation
// can be intercepted with `useBlocker` — e.g. to confirm before discarding
// unsaved workflow-graph edits. App reads `location` directly, so a single
// splat route renders the whole SPA.
//
// `errorElement` replaces React Router's bare "Unexpected Application Error!"
// overlay with a styled screen covering 404s and unreachable-backend
// (`Failed to fetch`) failures that escape the in-app ErrorBoundary.
const router = createBrowserRouter([
  { path: "*", element: <App />, errorElement: <RouteErrorBoundary /> },
]);

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Root element #root not found");
}

async function enableMocking() {
  if (!__USE_MOCK__) {
    return;
  }

  const { start } = await import("../mocks/browser");

  // `start()` returns a Promise that resolves
  // once the Service Worker is up and ready to intercept requests.
  return start();
}

enableMocking().then(() => {
  // Service worker (in dev:mock mode) is now in control of the page —
  // safe to fire plugin discovery without racing MSW activation.
  bootPlugins();

  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <RouterProvider router={router} />
    </React.StrictMode>,
  );
});
