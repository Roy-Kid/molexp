import { AlertTriangle, Info, XCircle } from "lucide-react";
import type { ConsoleEntry, WorkspaceSnapshot } from "@/app/types";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ExecutionConsoleProps {
  snapshot: WorkspaceSnapshot;
}

const levelStyles: Record<ConsoleEntry["level"], string> = {
  info: "bg-slate-100 text-slate-900",
  warning: "bg-amber-100 text-amber-900",
  error: "bg-rose-100 text-rose-900",
};

const levelIcons: Record<ConsoleEntry["level"], JSX.Element> = {
  info: <Info className="h-3.5 w-3.5" />,
  warning: <AlertTriangle className="h-3.5 w-3.5" />,
  error: <XCircle className="h-3.5 w-3.5" />,
};

export const ExecutionConsole = ({ snapshot }: ExecutionConsoleProps): JSX.Element => {
  return (
    <section className="h-28 border-t border-border bg-muted/30">
      <div className="flex items-center justify-between px-4 py-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Execution Console
        </p>
        <Badge variant="secondary" className="text-xs">
          {snapshot.consoleEntries.length} entries
        </Badge>
      </div>
      <ScrollArea className="h-20 px-4 pb-3">
        <div className="space-y-2">
          {snapshot.consoleEntries.length === 0 && (
            <p className="text-xs text-muted-foreground">No execution logs available.</p>
          )}
          {snapshot.consoleEntries.map((entry) => (
            <div
              key={entry.id}
              className="flex items-start gap-3 rounded-md border border-border/60 bg-background px-3 py-2"
            >
              <div className="mt-0.5 text-muted-foreground">{levelIcons[entry.level]}</div>
              <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2">
                  <Badge className={`${levelStyles[entry.level]} text-xs uppercase`}>
                    {entry.level}
                  </Badge>
                  <span className="text-xs text-muted-foreground">{entry.timestamp}</span>
                </div>
                <p className="text-xs text-foreground">{entry.message}</p>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </section>
  );
};
