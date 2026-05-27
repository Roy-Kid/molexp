import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { bootPlugins } from "@/plugins/runtime";
import App from "./App";
import "./styles/tailwind.css";
// xyflow's stylesheet is loaded once at the app entry so individual
// renderer modules can stay CSS-free — this matters for the node-side
// test runner which does not understand .css imports.
import "@xyflow/react/dist/style.css";

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
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </React.StrictMode>,
  );
});
