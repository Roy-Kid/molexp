import { lazy } from "react";

/** Sole Monaco entry for apps/web. Capability vendor stays inside this plugin. */
export const MonacoEditor = lazy(() => import("@monaco-editor/react"));
