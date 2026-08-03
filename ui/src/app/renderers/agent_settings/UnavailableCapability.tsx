import { AlertCircle, RefreshCw } from "lucide-react";
import type { JSX } from "react";
import { WorkbenchAction } from "@/components/workbench";

export const UnavailableCapability = ({
  title,
  description,
  onRetry,
}: {
  title: string;
  description: string;
  onRetry?: () => void;
}): JSX.Element => (
  <div
    role="status"
    className="flex items-start gap-3 border-y border-dashed border-border bg-surface-subtle px-4 py-4"
  >
    <span className="flex size-8 shrink-0 items-center justify-center text-muted-foreground">
      <AlertCircle className="size-4" />
    </span>
    <div className="min-w-0 flex-1">
      <p className="text-body font-medium text-foreground">{title}</p>
      <p className="mt-1 text-label leading-5 text-muted-foreground">{description}</p>
    </div>
    {onRetry && (
      <WorkbenchAction kind="secondary" size="compact" onClick={onRetry}>
        <RefreshCw className="mr-1 size-3.5" /> Retry
      </WorkbenchAction>
    )}
  </div>
);
