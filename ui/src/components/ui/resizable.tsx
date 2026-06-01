import { GripVerticalIcon } from "lucide-react";
import type * as React from "react";
import * as ResizablePrimitive from "react-resizable-panels";

import { cn } from "@/lib/utils";

// react-resizable-panels v4 renamed the primitives: PanelGroup -> Group,
// PanelResizeHandle -> Separator, and the group orientation prop direction ->
// orientation. v4 also drops the `data-panel-group-direction` attribute, so the
// old shadcn `data-[panel-group-direction=vertical]:*` style hooks no longer
// match. And layout persistence moved from the `autoSaveId` prop to the
// `useDefaultLayout` hook. These wrappers keep the shadcn call sites unchanged:
// they still accept `direction` and `autoSaveId`, translating both to the v4
// API; the group manages flex-direction internally for vertical layouts.

function ResizablePanelGroup({
  className,
  direction,
  autoSaveId,
  ...props
}: Omit<React.ComponentProps<typeof ResizablePrimitive.Group>, "orientation"> & {
  direction?: ResizablePrimitive.Orientation;
  autoSaveId?: string;
}) {
  const persisted = ResizablePrimitive.useDefaultLayout(
    autoSaveId
      ? { id: autoSaveId, storage: globalThis.localStorage }
      : // No persistence requested: a stable unused id keeps hook order stable.
        { id: "resizable-ephemeral" },
  );

  return (
    <ResizablePrimitive.Group
      data-slot="resizable-panel-group"
      orientation={direction}
      className={cn("flex h-full w-full", className)}
      {...(autoSaveId
        ? { defaultLayout: persisted.defaultLayout, onLayoutChanged: persisted.onLayoutChanged }
        : {})}
      {...props}
    />
  );
}

function ResizablePanel({ ...props }: React.ComponentProps<typeof ResizablePrimitive.Panel>) {
  return <ResizablePrimitive.Panel data-slot="resizable-panel" {...props} />;
}

function ResizableHandle({
  withHandle,
  className,
  ...props
}: React.ComponentProps<typeof ResizablePrimitive.Separator> & {
  withHandle?: boolean;
}) {
  return (
    <ResizablePrimitive.Separator
      data-slot="resizable-handle"
      className={cn(
        "bg-border focus-visible:ring-ring relative flex w-px items-center justify-center after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2 focus-visible:ring-1 focus-visible:ring-offset-1 focus-visible:outline-hidden",
        className,
      )}
      {...props}
    >
      {withHandle && (
        <div className="bg-border z-10 flex h-4 w-3 items-center justify-center rounded-xs border">
          <GripVerticalIcon className="size-2.5" />
        </div>
      )}
    </ResizablePrimitive.Separator>
  );
}

export { ResizableHandle, ResizablePanel, ResizablePanelGroup };
