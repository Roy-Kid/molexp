import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Card, CardHeader } from '@/components/ui/card';
import { Play } from 'lucide-react';
import { cn } from '@/lib/utils';

export const StartNode = ({ selected }: NodeProps) => {
  return (
    <Card className={cn(
      "w-[180px] border-green-200 bg-green-50/50 shadow-sm",
      selected ? "ring-2 ring-green-500 ring-offset-2" : ""
    )}>
      <CardHeader className="p-3 flex flex-row items-center gap-3 space-y-0">
        <div className="p-2 rounded-full bg-green-100 text-green-600">
          <Play className="w-4 h-4 fill-current" />
        </div>
        <span className="font-semibold text-green-900">Start</span>
      </CardHeader>
      <Handle type="source" position={Position.Bottom} className="w-3 h-3 bg-green-500" />
    </Card>
  );
};
