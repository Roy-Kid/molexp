import { cn } from "@/lib/utils";

function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="skeleton"
      className={cn("mol-motion-progress-pulse rounded-control bg-interactive", className)}
      {...props}
    />
  );
}

export { Skeleton };
