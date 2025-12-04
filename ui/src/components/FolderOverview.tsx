import React, { useState, useEffect } from 'react';
import { File, Folder, Loader } from 'lucide-react';
import { ExplorerHeader } from './ExplorerHeader';
import { API_ENDPOINTS } from '@/config/api';


interface FolderOverviewProps {
  folderId: string;
  path: string;
  name: string;
  onNavigate: (path: string, type: 'file' | 'folder') => void;
  onToggleDetails: () => void;
}

interface FolderEntry {
  name: string;
  path: string;
  type: 'directory' | 'file';
  size?: number;
}

export const FolderOverview: React.FC<FolderOverviewProps> = ({
  folderId,
  path,
  name,
  onNavigate,
}) => {
  const [entries, setEntries] = useState<FolderEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    
    const fetchContents = async () => {
      console.log('[FolderOverview] Fetching contents for:', folderId, path);
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(API_ENDPOINTS.workspace.folders.browse(folderId, path));
        if (!response.ok) throw new Error('Failed to load folder contents');
        const data = await response.json();
        if (isMounted) {
          setEntries(data.entries);
        }
      } catch (err) {
        if (isMounted) {
          setError('Error loading folder contents');
          console.error(err);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchContents();
    
    return () => {
      isMounted = false;
    };
  }, [folderId, path]); // Only re-fetch when these actually change

  const formatSize = (bytes?: number) => {
    if (bytes === undefined) return '-';
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const stats = {
    files: entries.filter(e => e.type === 'file').length,
    folders: entries.filter(e => e.type === 'directory').length,
    totalSize: entries.reduce((acc, e) => acc + (e.size || 0), 0),
  };

  return (
    <div className="flex flex-col h-full bg-background">
      <ExplorerHeader
        name={name}
        path={path || '/'}
        type="folder"
      />

      <div className="flex-1 overflow-auto p-6">
        {loading ? (
          <div className="flex justify-center py-10">
            <Loader className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="text-red-500 text-center py-10">{error}</div>
        ) : (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 border rounded-lg bg-card">
                <div className="text-2xl font-bold">{entries.length}</div>
                <div className="text-xs text-muted-foreground">Total Items</div>
              </div>
              <div className="p-4 border rounded-lg bg-card">
                <div className="text-2xl font-bold">{stats.folders}</div>
                <div className="text-xs text-muted-foreground">Folders</div>
              </div>
              <div className="p-4 border rounded-lg bg-card">
                <div className="text-2xl font-bold">{formatSize(stats.totalSize)}</div>
                <div className="text-xs text-muted-foreground">Total Size</div>
              </div>
            </div>

            {/* Content List */}
            <div className="border rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-muted/50">
                  <tr className="text-left text-muted-foreground">
                    <th className="p-3 font-medium">Name</th>
                    <th className="p-3 font-medium w-24">Size</th>
                    <th className="p-3 font-medium w-24">Type</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {entries.length === 0 ? (
                    <tr>
                      <td colSpan={3} className="p-8 text-center text-muted-foreground">
                        Empty folder
                      </td>
                    </tr>
                  ) : (
                    entries.map((entry) => (
                      <tr
                        key={entry.path}
                        className="hover:bg-accent/50 cursor-pointer transition-colors"
                        onClick={() => onNavigate(entry.path, entry.type === 'directory' ? 'folder' : 'file')}
                      >
                        <td className="p-3 flex items-center gap-2">
                          {entry.type === 'directory' ? (
                            <Folder className="h-4 w-4 text-blue-500" />
                          ) : (
                            <File className="h-4 w-4 text-gray-500" />
                          )}
                          <span className="truncate">{entry.name}</span>
                        </td>
                        <td className="p-3 text-muted-foreground font-mono text-xs">
                          {formatSize(entry.size)}
                        </td>
                        <td className="p-3 text-muted-foreground text-xs capitalize">
                          {entry.type === 'directory' ? 'Folder' : 'File'}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
