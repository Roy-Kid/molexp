import { ArrowLeft, FileQuestion, RefreshCw, ServerCrash, WifiOff } from "lucide-react";
import type { ComponentType, ReactNode } from "react";
import { isRouteErrorResponse, useRouteError } from "react-router-dom";
import { Button } from "@/components/ui/button";

/**
 * Route-level error element for the data router. React Router renders this in
 * place of its bare-bones default overlay ("Unexpected Application Error!")
 * whenever a render, loader, or action throws — including the common
 * `TypeError: Failed to fetch` raised when the backend is unreachable.
 *
 * The inner `app/layout/ErrorBoundary` still guards normal in-app render
 * errors; this one is the last line of defense at the router boundary and also
 * covers 404-style route responses.
 */

interface ErrorShape {
  /** Big, glanceable label — the HTTP status or a short tag. */
  badge: string;
  Icon: ComponentType<{ className?: string }>;
  title: string;
  description: string;
  /** Technical detail shown in the collapsible developer section. */
  detail?: string;
  /** A failed fetch is usually transient — surface a prominent retry. */
  retryable: boolean;
}

function classify(error: unknown): ErrorShape {
  if (isRouteErrorResponse(error)) {
    if (error.status === 404) {
      return {
        badge: "404",
        Icon: FileQuestion,
        title: "Page not found",
        description:
          "We couldn't find that page. It may have been moved, or the link is out of date.",
        detail: error.data ? String(error.data) : undefined,
        retryable: false,
      };
    }
    return {
      badge: String(error.status),
      Icon: ServerCrash,
      title: error.statusText || "Request failed",
      description:
        "The server responded with an error. Try again in a moment, or head back to your workspace.",
      detail: error.data ? String(error.data) : undefined,
      retryable: true,
    };
  }

  const message = error instanceof Error ? error.message : typeof error === "string" ? error : "";

  // `TypeError: Failed to fetch` is what the browser throws when the request
  // never reaches the backend — server down, wrong port, or no network.
  if (/failed to fetch|networkerror|load failed/i.test(message)) {
    return {
      badge: "Offline",
      Icon: WifiOff,
      title: "Can't reach the server",
      description:
        "The molexp backend didn't respond. Check that the server is running and reachable, then try again.",
      detail: error instanceof Error ? `${error.name}: ${error.message}` : message,
      retryable: true,
    };
  }

  return {
    badge: "Error",
    Icon: ServerCrash,
    title: "Something went wrong",
    description: "An unexpected error stopped the page from rendering. Reloading often clears it.",
    detail:
      error instanceof Error ? (error.stack ?? `${error.name}: ${error.message}`) : String(error),
    retryable: true,
  };
}

export function RouteErrorBoundary(): ReactNode {
  const error = useRouteError();
  const { badge, Icon, title, description, detail, retryable } = classify(error);

  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden bg-background px-6 py-12">
      {/* Soft radial backdrop so the card reads as the focal point. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_50%_at_50%_0%,hsl(var(--primary)/0.08),transparent_70%)]"
      />

      <div className="relative w-full max-w-md text-center">
        <div className="mb-6 flex justify-center">
          <div className="relative">
            <div className="absolute inset-0 rounded-2xl bg-destructive/15 blur-xl" />
            <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl border border-destructive/20 bg-destructive/10 text-destructive">
              <Icon className="h-8 w-8" />
            </div>
          </div>
        </div>

        <p className="mb-2 font-mono text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
          {badge}
        </p>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">{title}</h1>
        <p className="mx-auto mt-3 max-w-sm text-sm leading-relaxed text-muted-foreground">
          {description}
        </p>

        <div className="mt-7 flex flex-col items-center justify-center gap-3 sm:flex-row">
          {retryable && (
            <Button onClick={() => window.location.reload()} className="w-full sm:w-auto">
              <RefreshCw className="h-4 w-4" />
              Try again
            </Button>
          )}
          <Button
            variant="outline"
            onClick={() => {
              window.location.href = "/";
            }}
            className="w-full sm:w-auto"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to workspace
          </Button>
        </div>

        {detail && (
          <details className="group mt-8 text-left">
            <summary className="cursor-pointer list-none text-center text-xs font-medium text-muted-foreground transition-colors hover:text-foreground">
              <span className="group-open:hidden">Show technical details</span>
              <span className="hidden group-open:inline">Hide technical details</span>
            </summary>
            <pre className="mt-3 max-h-56 overflow-auto rounded-md border border-border bg-muted/40 p-3 text-left font-mono text-xs leading-relaxed text-foreground/80">
              {detail}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}
