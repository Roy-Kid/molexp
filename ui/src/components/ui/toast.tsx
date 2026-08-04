/**
 * Status-bar façade — same call sites as a toast API, but every message
 * lands in the bottom workbench status strip (MolVis-aligned).
 *
 * Call `toast("Copied")` or `toast.error("…")` from anywhere. There is no
 * floating card host; {@link ToastProvider} is a passthrough for legacy mounts.
 */

import { createContext, type ReactNode, useContext, useMemo } from "react";

import { reportStatus, type StatusReportType } from "@/lib/status-report";

export type ToastKind = "default" | "success" | "error";

export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface ToastOptions {
  kind?: ToastKind;
  durationMs?: number;
  /** Ignored — status bar is one-line; keep for call-site compatibility. */
  action?: ToastAction;
}

type ToastFn = ((message: string, options?: ToastOptions) => void) & {
  success: (message: string, options?: Omit<ToastOptions, "kind">) => void;
  error: (message: string, options?: Omit<ToastOptions, "kind">) => void;
};

function kindToType(kind: ToastKind | undefined): StatusReportType {
  if (kind === "error") return "error";
  if (kind === "success") return "success";
  return "info";
}

function push(message: string, options?: ToastOptions): void {
  const text = message.trim();
  if (!text) return;
  reportStatus(text, kindToType(options?.kind));
}

/** Imperative API (usable outside React trees). Routes to the status bus. */
export const toast: ToastFn = Object.assign(
  (message: string, options?: ToastOptions) => {
    push(message, options);
  },
  {
    success: (message: string, options?: Omit<ToastOptions, "kind">) => {
      push(message, { ...options, kind: "success" });
    },
    error: (message: string, options?: Omit<ToastOptions, "kind">) => {
      push(message, { ...options, kind: "error" });
    },
  },
);

const ToastContext = createContext<ToastFn | null>(null);

export function useToast(): ToastFn {
  return useContext(ToastContext) ?? toast;
}

/**
 * Passthrough provider — no floating toast host.
 * Kept so root mounts and tests that wrap with ToastProvider still compile.
 */
export function ToastProvider({ children }: { children: ReactNode }): JSX.Element {
  const api = useMemo<ToastFn>(() => toast, []);
  return <ToastContext.Provider value={api}>{children}</ToastContext.Provider>;
}
