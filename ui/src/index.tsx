import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
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
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </React.StrictMode>,
  );
});
