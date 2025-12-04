import React, { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { API_ENDPOINTS } from '@/config/api';

interface AddWorkspaceFolderDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onFolderAdded: () => void;
}

export const AddWorkspaceFolderDialog: React.FC<AddWorkspaceFolderDialogProps> = ({
  open,
  onOpenChange,
  onFolderAdded,
}) => {
  const [path, setPath] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!path.trim()) {
      setError('Path is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(API_ENDPOINTS.workspace.folders.add, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          path: path.trim(),
          name: name.trim() || undefined,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to add folder');
      }

      // Success
      setPath('');
      setName('');
      onOpenChange(false);
      onFolderAdded();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add Folder to Workspace</DialogTitle>
          <DialogDescription>
            Add a directory to browse its contents in the workspace explorer.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="path">Folder Path *</Label>
              <Input
                id="path"
                placeholder="/path/to/folder"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                disabled={loading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="name">Display Name (optional)</Label>
              <Input
                id="name"
                placeholder="My Folder"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={loading}
              />
            </div>

            {error && (
              <div className="text-sm text-red-500">
                {error}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={loading}>
              {loading ? 'Adding...' : 'Add Folder'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
