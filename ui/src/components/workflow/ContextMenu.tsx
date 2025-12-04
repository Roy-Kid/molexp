
import { Button } from '../ui/button';

interface ContextMenuProps {
  id: string;
  top: number;
  left: number;
  right?: number;
  bottom?: number;
  onEdit?: () => void;
  onDelete?: () => void;
  onViewResults?: () => void;
  onToggleOutput?: () => void;
  isOutput?: boolean;
  // ... other props
}

export const ContextMenu = ({ id, top, left, right, bottom, onEdit, onDelete, onViewResults, onToggleOutput, isOutput, ...props }: ContextMenuProps) => {
  return (
    <div
      style={{ top, left, right, bottom }}
      className="absolute z-10 bg-white border rounded shadow-md p-2 flex flex-col gap-1 w-32"
      {...props}
    >
      {onEdit && (
        <Button variant="ghost" size="sm" className="justify-start" onClick={onEdit}>
          Edit
        </Button>
      )}
      {onDelete && (
        <Button variant="ghost" size="sm" className="justify-start text-red-600 hover:text-red-700 hover:bg-red-50" onClick={onDelete}>
          Delete
        </Button>
      )}
      {onViewResults && (
        <Button variant="ghost" size="sm" className="justify-start text-blue-600 hover:text-blue-700 hover:bg-blue-50" onClick={onViewResults}>
          View Results
        </Button>
      )}
      {onToggleOutput && (
        <Button variant="ghost" size="sm" className="justify-start" onClick={onToggleOutput}>
          {isOutput ? 'Unset Output' : 'Set as Output'}
        </Button>
      )}
    </div>
  );
};
