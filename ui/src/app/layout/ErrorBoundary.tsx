import { AlertTriangle } from "lucide-react";
import type { ErrorInfo, ReactNode } from "react";
import { Component } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

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
        <Card className="w-full max-w-lg border-destructive/30">
          <CardHeader className="space-y-2">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive/10 text-destructive">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <CardTitle className="text-lg font-semibold text-destructive">
                Execution halted
              </CardTitle>
            </div>
            <p className="text-sm text-muted-foreground">
              A runtime error stopped the UI. Resolve the issue and reload to continue.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            {this.state.error && (
              <div className="rounded-md border border-border bg-muted/40 p-3 text-xs text-foreground">
                {this.state.error.message}
              </div>
            )}
            <Button variant="outline" onClick={this.handleReload}>
              Reload Workspace
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }
}
