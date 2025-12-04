import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity, CheckCircle2, XCircle, Clock, PlayCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

export const ProcessNode = ({ data, selected }: NodeProps) => {
  const status = data.status as string | undefined;
  const plannedColor = data.plannedColor as string | undefined;
  const isOutput = data.isOutput as boolean | undefined;

  const getStatusColor = () => {
    if (isOutput) return 'border-blue-400 bg-blue-50/80 ring-2 ring-blue-200 shadow-blue-100';
    
    switch (status) {
      case 'success': return 'border-green-500 bg-green-50/50';
      case 'failed': return 'border-red-500 bg-red-50/50';
      case 'running': return 'border-blue-500 bg-blue-50/50 ring-2 ring-blue-200';
      case 'pending': return 'border-yellow-500 bg-yellow-50/50';
      case 'planned': 
        // We use style prop for dynamic color, but set base classes here
        return 'bg-card ring-2 ring-offset-1 dashed border-2'; 
      default: return 'border-border bg-card hover:border-slate-400';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'success': return <CheckCircle2 className="w-3.5 h-3.5 text-green-600" />;
      case 'failed': return <XCircle className="w-3.5 h-3.5 text-red-600" />;
      case 'running': return <Activity className="w-3.5 h-3.5 text-blue-600 animate-spin" />;
      case 'pending': return <Clock className="w-3.5 h-3.5 text-yellow-600" />;
      case 'planned': return <Clock className="w-3.5 h-3.5" style={{ color: plannedColor }} />;
      default: return <PlayCircle className="w-3.5 h-3.5 text-muted-foreground" />;
    }
  };

  return (
    <Card 
      className={cn(
        "w-[240px] h-auto shadow-sm transition-all duration-200 flex flex-col gap-0 p-0", // Override Card defaults
        getStatusColor(),
        selected ? "ring-2 ring-primary ring-offset-1" : ""
      )}
      style={status === 'planned' && plannedColor ? { 
        borderColor: plannedColor,
        '--tw-ring-color': plannedColor + '40' // 25% opacity for ring
      } as React.CSSProperties : undefined}
    >
      <Handle type="target" position={Position.Top} className="w-2.5 h-2.5 bg-slate-400 hover:bg-slate-600 transition-colors" />
      
      <CardHeader className="p-2.5 pb-1.5 space-y-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 overflow-hidden">
            <div className={cn("p-1 rounded-md bg-background/80 border shadow-sm shrink-0", isOutput ? "text-blue-600" : "")}>
              {getStatusIcon()}
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider truncate">
                {data.category as string || 'Node'}
              </span>
              <span className="font-semibold text-xs leading-tight truncate" title={data.label as string}>
                {data.label as string}
              </span>
            </div>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-2.5 pt-1.5 flex-1 flex flex-col justify-between">
        {(data.description as string) && (
          <p className="text-[10px] text-muted-foreground leading-snug break-words">
            {(data.description as string)}
          </p>
        )}
        {status && (
          <div className="mt-auto pt-2 flex justify-end">
             <Badge variant="outline" className={cn(
               "text-[9px] px-1 py-0 h-4 uppercase tracking-wide",
               status === 'success' && "border-green-200 text-green-700 bg-green-50",
               status === 'failed' && "border-red-200 text-red-700 bg-red-50",
               status === 'running' && "border-blue-200 text-blue-700 bg-blue-50",
               status === 'planned' && "bg-opacity-10",
             )}
             style={status === 'planned' && plannedColor ? {
               borderColor: plannedColor,
               color: plannedColor,
               backgroundColor: plannedColor + '20'
             } : undefined}
             >
               {status}
             </Badge>
          </div>
        )}
      </CardContent>

      <Handle type="source" position={Position.Bottom} className="w-2.5 h-2.5 bg-slate-400 hover:bg-slate-600 transition-colors" />
    </Card>
  );
};
