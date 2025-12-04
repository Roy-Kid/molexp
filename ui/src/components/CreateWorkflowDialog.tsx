import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { API_ENDPOINTS } from '@/config/api';

interface CreateWorkflowDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (filename: string) => void;
  folderId: string;
  basePath?: string;
}

export const CreateWorkflowDialog: React.FC<CreateWorkflowDialogProps> = ({
  isOpen,
  onClose,
  onSuccess,
  folderId,
  basePath = '',
}) => {
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      let filename = name;
      if (filename.endsWith('.flow')) {
        filename = filename.slice(0, -5);
      }
      filename += '.flow';
      
      const fullPath = basePath ? `${basePath}/${filename}` : filename;

      // Create empty workflow file
      const emptyWorkflow = {
        nodes: [],
        edges: [],
        description: '',
        name: name.replace('.flow', ''),
      };

      const response = await fetch(API_ENDPOINTS.workspace.files.write, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          folder_id: folderId,
          path: fullPath,
          content: JSON.stringify(emptyWorkflow, null, 2),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create workflow file');
      }

      onSuccess(`workspace:${fullPath}`);
      onClose();
      setName('');
    } catch (err) {
      console.error(err);
      alert('Failed to create workflow');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create New Workflow</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Workflow Name</Label>
            <div className="flex items-center gap-2">
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="my-workflow"
                required
                className="flex-1"
              />
              <span className="text-sm text-muted-foreground font-mono">.flow</span>
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={loading}>
              {loading ? 'Creating...' : 'Create Workflow'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
