import { ChevronDown } from "lucide-react";
import { Collapsible as CollapsiblePrimitive } from "radix-ui";
import type * as React from "react";

import { cn } from "@/lib/utils";

function Collapsible(props: React.ComponentProps<typeof CollapsiblePrimitive.Root>) {
  return <CollapsiblePrimitive.Root data-slot="collapsible" {...props} />;
}

function CollapsibleTrigger({
  className,
  children,
  ...props
}: React.ComponentProps<typeof CollapsiblePrimitive.Trigger>) {
  return (
    <CollapsiblePrimitive.Trigger
      data-slot="collapsible-trigger"
      className={cn(
        "group/collapsible inline-flex items-center gap-2 rounded-control text-body font-medium outline-none transition-colors duration-(--motion-fast) ease-standard hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50",
        className,
      )}
      {...props}
    >
      {children}
      <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform duration-(--motion-fast) ease-standard group-data-[state=open]/collapsible:rotate-180" />
    </CollapsiblePrimitive.Trigger>
  );
}

function CollapsibleContent({
  className,
  ...props
}: React.ComponentProps<typeof CollapsiblePrimitive.Content>) {
  return (
    <CollapsiblePrimitive.Content
      data-slot="collapsible-content"
      className={cn(
        "overflow-hidden data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down",
        className,
      )}
      {...props}
    />
  );
}

export { Collapsible, CollapsibleContent, CollapsibleTrigger };
