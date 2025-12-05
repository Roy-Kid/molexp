import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  private handleReload = () => {
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-background p-4">
          <Card className="w-full max-w-md border-destructive/20 shadow-lg">
            <CardHeader className="flex flex-row items-center gap-4 space-y-0 pb-2">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-destructive/10">
                <AlertTriangle className="h-6 w-6 text-destructive" />
              </div>
              <div>
                <CardTitle className="text-xl font-bold text-destructive">
                  Something went wrong
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-4">
              <p className="text-sm text-muted-foreground">
                An unexpected error occurred while rendering this page.
              </p>
              {this.state.error && (
                <div className="mt-4 rounded-md bg-muted p-3">
                  <p className="font-mono text-xs text-foreground break-all">
                    {this.state.error.message}
                  </p>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-end pt-2">
              <Button 
                variant="outline" 
                onClick={this.handleReload}
                className="gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Reload Page
              </Button>
            </CardFooter>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}
