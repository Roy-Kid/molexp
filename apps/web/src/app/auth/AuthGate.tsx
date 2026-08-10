import type { ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "@/app/auth/AuthContext";

/**
 * When the server has auth enabled, require a session before rendering children.
 * Redirects to ``/login?next=…`` otherwise.
 */
export function AuthGate({ children }: { children: ReactNode }): JSX.Element {
  const { loading, enabled, authenticated } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background text-body text-muted-foreground">
        Checking session…
      </div>
    );
  }

  if (enabled && !authenticated) {
    const next = `${location.pathname}${location.search}`;
    return <Navigate to={`/login?next=${encodeURIComponent(next)}`} replace />;
  }

  return <>{children}</>;
}
