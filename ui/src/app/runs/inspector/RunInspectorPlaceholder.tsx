import type { JSX, ReactNode } from "react";

interface RunInspectorPlaceholderProps {
  title: string;
  description: ReactNode;
}

export const RunInspectorPlaceholder = ({
  title,
  description,
}: RunInspectorPlaceholderProps): JSX.Element => (
  <div className="flex h-full flex-col items-start justify-start gap-1 px-4 py-4 text-xs text-muted-foreground">
    <p className="text-foreground">{title}</p>
    <p>{description}</p>
  </div>
);
