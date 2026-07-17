import { AlertCircle, RefreshCw } from "lucide-react";
import type { JSX } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export const UnavailableCapability = ({
  title,
  description,
  onRetry,
}: {
  title: string;
  description: string;
  onRetry?: () => void;
}): JSX.Element => (
  <Card className="border-dashed border-border/80 bg-muted/15">
    <CardContent className="flex items-start gap-3 p-4">
      <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground">
        <AlertCircle className="size-4" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium">{title}</p>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{description}</p>
      </div>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="mr-1 size-3.5" /> Retry
        </Button>
      )}
    </CardContent>
  </Card>
);
