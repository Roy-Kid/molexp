import React from 'react';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface DetailOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

export const DetailOverlay: React.FC<DetailOverlayProps> = ({
  isOpen,
  onClose,
  children,
}) => {
  return (
    <div
      className={cn(
        "absolute top-0 right-0 bottom-0 w-96 bg-background border-l shadow-xl transform transition-transform duration-200 ease-in-out z-20 flex flex-col",
        isOpen ? "translate-x-0" : "translate-x-full"
      )}
    >
      <div className="flex items-center justify-between p-4 border-b">
        <h3 className="font-semibold">Details</h3>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex-1 overflow-hidden">
        {children}
      </div>
    </div>
  );
};
