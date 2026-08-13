import { AlertTriangle } from "lucide-react";
import type { ErrorInfo, ReactNode } from "react";
import { Component } from "react";
import { WorkbenchAction } from "@/components/workbench";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public state: ErrorBoundaryState = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("Unhandled error:", error, errorInfo);
  }

  private handleReload = (): void => {
    window.location.reload();
  };

  public render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-6">
        <section
          role="alert"
          className="w-full max-w-lg space-y-3 border-y border-status-failed/30 bg-status-failed-soft px-4 py-4"
        >
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-control bg-status-failed-soft text-status-failed-foreground">
              <AlertTriangle className="h-4 w-4" />
            </div>
            <h1 className="text-title font-semibold text-status-failed-foreground">
              Execution halted
            </h1>
          </div>
          <p className="text-body text-muted-foreground">
            A runtime error stopped the UI. Resolve the issue and reload to continue.
          </p>
          {this.state.error && (
            <pre className="overflow-auto rounded-control border border-border bg-muted/40 p-3 font-mono text-micro text-foreground">
              {this.state.error.message}
            </pre>
          )}
          <WorkbenchAction kind="secondary" onClick={this.handleReload}>
            Reload Workspace
          </WorkbenchAction>
        </section>
      </div>
    );
  }
}
