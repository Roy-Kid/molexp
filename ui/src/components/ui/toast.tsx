/**
 * Minimal toast host — no third-party dependency.
 * Call `toast("Copied")` or `toast.error("…")` from anywhere.
 */

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { cn } from "@/lib/utils";

export type ToastKind = "default" | "success" | "error";

export interface ToastAction {
  label: string;
  onClick: () => void;
}

export interface ToastOptions {
  kind?: ToastKind;
  durationMs?: number;
  action?: ToastAction;
}

interface ToastItem extends ToastOptions {
  id: number;
  message: string;
}

type ToastFn = ((message: string, options?: ToastOptions) => void) & {
  success: (message: string, options?: Omit<ToastOptions, "kind">) => void;
  error: (message: string, options?: Omit<ToastOptions, "kind">) => void;
};

let pushExternal: ((message: string, options?: ToastOptions) => void) | null = null;

/** Imperative API (usable outside React trees). */
export const toast: ToastFn = Object.assign(
  (message: string, options?: ToastOptions) => {
    pushExternal?.(message, options);
  },
  {
    success: (message: string, options?: Omit<ToastOptions, "kind">) => {
      pushExternal?.(message, { ...options, kind: "success" });
    },
    error: (message: string, options?: Omit<ToastOptions, "kind">) => {
      pushExternal?.(message, { ...options, kind: "error" });
    },
  },
);

const ToastContext = createContext<ToastFn | null>(null);

export function useToast(): ToastFn {
  return useContext(ToastContext) ?? toast;
}

const KIND_CLASS: Record<ToastKind, string> = {
  default: "border-border bg-card text-foreground",
  success: "border-success/30 bg-success-soft text-success-foreground",
  error: "border-destructive/30 bg-destructive/10 text-destructive",
};

const DEFAULT_MS = 2800;

export function ToastProvider({ children }: { children: ReactNode }): JSX.Element {
  const [items, setItems] = useState<ToastItem[]>([]);

  const dismiss = useCallback((id: number) => {
    setItems((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (message: string, options?: ToastOptions) => {
      const id = Date.now() + Math.floor(Math.random() * 1000);
      const item: ToastItem = {
        id,
        message,
        kind: options?.kind ?? "default",
        durationMs: options?.durationMs ?? DEFAULT_MS,
        action: options?.action,
      };
      setItems((prev) => [...prev.slice(-3), item]);
      window.setTimeout(() => dismiss(id), item.durationMs ?? DEFAULT_MS);
    },
    [dismiss],
  );

  useEffect(() => {
    pushExternal = push;
    return () => {
      if (pushExternal === push) pushExternal = null;
    };
  }, [push]);

  const api = useMemo<ToastFn>(() => {
    const fn = ((message: string, options?: ToastOptions) => push(message, options)) as ToastFn;
    fn.success = (message, options) => push(message, { ...options, kind: "success" });
    fn.error = (message, options) => push(message, { ...options, kind: "error" });
    return fn;
  }, [push]);

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        className="pointer-events-none fixed bottom-4 right-4 z-[100] flex w-[min(100vw-2rem,20rem)] flex-col gap-2"
        aria-live="polite"
      >
        {items.map((item) => (
          <div
            key={item.id}
            className={cn(
              "pointer-events-auto flex items-center gap-2 rounded-md border px-3 py-2 text-sm shadow-md",
              KIND_CLASS[item.kind ?? "default"],
            )}
            role="status"
          >
            <span className="min-w-0 flex-1 truncate">{item.message}</span>
            {item.action && (
              <button
                type="button"
                className="flex-none text-xs font-medium underline-offset-2 hover:underline"
                onClick={() => {
                  item.action?.onClick();
                  dismiss(item.id);
                }}
              >
                {item.action.label}
              </button>
            )}
            <button
              type="button"
              className="flex-none text-xs text-muted-foreground hover:text-foreground"
              aria-label="Dismiss"
              onClick={() => dismiss(item.id)}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
