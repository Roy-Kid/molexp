import { AlertCircle } from "lucide-react";
import type { JSX } from "react";
import { WorkbenchRetryAction } from "@/components/workbench";

export const UnavailableCapability = ({
  title,
  description,
  onRetry,
}: {
  title: string;
  description: string;
  onRetry?: () => void;
}): JSX.Element => (
  <div role="status" className="flex items-start gap-3 bg-surface-subtle px-4 py-4">
    <span className="flex size-8 shrink-0 items-center justify-center text-muted-foreground">
      <AlertCircle className="size-4" />
    </span>
    <div className="min-w-0 flex-1">
      <p className="text-body font-medium text-foreground">{title}</p>
      <p className="mt-1 text-label leading-5 text-muted-foreground">{description}</p>
    </div>
    {onRetry && <WorkbenchRetryAction onClick={onRetry} />}
  </div>
);
