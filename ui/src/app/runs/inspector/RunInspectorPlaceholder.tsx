import type { JSX, ReactNode } from "react";

interface RunInspectorPlaceholderProps {
  title: string;
  description: ReactNode;
}

export const RunInspectorPlaceholder = ({
  title,
  description,
}: RunInspectorPlaceholderProps): JSX.Element => (
  <div className="flex h-full flex-col items-start justify-start gap-2 px-4 py-4 text-sm">
    <p className="font-medium text-foreground">{title}</p>
    <p className="text-xs leading-relaxed text-muted-foreground">{description}</p>
  </div>
);
