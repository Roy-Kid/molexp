import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Card, CardHeader } from '@/components/ui/card';
import { Square } from 'lucide-react';
import { cn } from '@/lib/utils';

export const EndNode = ({ selected }: NodeProps) => {
  return (
    <Card className={cn(
      "w-[180px] border-red-200 bg-red-50/50 shadow-sm",
      selected ? "ring-2 ring-red-500 ring-offset-2" : ""
    )}>
      <Handle type="target" position={Position.Top} className="w-3 h-3 bg-red-500" />
      <CardHeader className="p-3 flex flex-row items-center gap-3 space-y-0">
        <div className="p-2 rounded-full bg-red-100 text-red-600">
          <Square className="w-4 h-4 fill-current" />
        </div>
        <span className="font-semibold text-red-900">End</span>
      </CardHeader>
    </Card>
  );
};
