import { useState } from 'react';
import { Folder, File, FileText, Database, ChevronRight, ChevronDown, Search } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';

// Mock Data
const fileSystem = [
  {
    id: 'root',
    name: 'My Projects',
    type: 'folder',
    children: [
      {
        id: 'proj1',
        name: 'Aspirin Study',
        type: 'folder',
        children: [
          { id: 'f1', name: 'aspirin.pdb', type: 'file', fileType: 'pdb', size: '12 KB', date: '2023-10-25' },
          { id: 'f2', name: 'optimization.log', type: 'file', fileType: 'log', size: '45 KB', date: '2023-10-26' },
          { id: 'f3', name: 'results.json', type: 'file', fileType: 'json', size: '2 KB', date: '2023-10-27' },
        ]
      },
      {
        id: 'proj2',
        name: 'Protein Binding',
        type: 'folder',
        children: [
          { id: 'f4', name: 'protein.pdb', type: 'file', fileType: 'pdb', size: '2.4 MB', date: '2023-10-20' },
          { id: 'f5', name: 'ligand.sdf', type: 'file', fileType: 'sdf', size: '5 KB', date: '2023-10-21' },
        ]
      },
      { id: 'f6', name: 'notes.txt', type: 'file', fileType: 'txt', size: '1 KB', date: '2023-10-01' },
    ]
  }
];

const FileIcon = ({ type }: { type: string }) => {
  switch (type) {
    case 'pdb':
    case 'sdf':
      return <Database className="h-8 w-8 text-blue-500" />;
    case 'log':
    case 'txt':
      return <FileText className="h-8 w-8 text-gray-500" />;
    case 'json':
      return <File className="h-8 w-8 text-yellow-500" />;
    default:
      return <File className="h-8 w-8 text-gray-400" />;
  }
};

export const Assets = () => {
  const [currentFolder, setCurrentFolder] = useState<any>(fileSystem[0]);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['root']));

  const toggleFolder = (folderId: string) => {
    const newExpanded = new Set(expandedFolders);
    if (newExpanded.has(folderId)) {
      newExpanded.delete(folderId);
    } else {
      newExpanded.add(folderId);
    }
    setExpandedFolders(newExpanded);
  };

  const navigateTo = (folder: any) => {
    setCurrentFolder(folder);
    // Ideally update path breadcrumbs here
  };

  const renderTree = (items: any[], level = 0) => {
    return items.map(item => (
      <div key={item.id} style={{ paddingLeft: `${level * 12}px` }}>
        {item.type === 'folder' ? (
          <div>
            <div 
              className={`flex items-center py-1 px-2 rounded cursor-pointer hover:bg-accent ${currentFolder.id === item.id ? 'bg-accent' : ''}`}
              onClick={() => navigateTo(item)}
            >
              <button 
                onClick={(e) => { e.stopPropagation(); toggleFolder(item.id); }} 
                className="p-1 hover:bg-muted rounded mr-1"
              >
                {expandedFolders.has(item.id) ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              </button>
              <Folder className="h-4 w-4 mr-2 text-blue-500" />
              <span className="text-sm truncate">{item.name}</span>
            </div>
            {expandedFolders.has(item.id) && item.children && (
              <div>{renderTree(item.children, level + 1)}</div>
            )}
          </div>
        ) : null}
      </div>
    ));
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-6 py-4 border-b">
        <h2 className="text-2xl font-bold tracking-tight">Assets</h2>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search assets..." className="pl-8 w-64" />
          </div>
          <Button>Upload</Button>
        </div>
      </div>
      
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar Tree */}
        <div className="w-64 border-r bg-muted/10 overflow-y-auto p-2">
          {renderTree(fileSystem)}
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="mb-4 text-sm text-muted-foreground">
            {currentFolder.name}
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {currentFolder.children?.map((item: any) => (
              <Card key={item.id} className="cursor-pointer hover:shadow-md transition-shadow">
                <CardContent className="p-4 flex flex-col items-center text-center gap-3">
                  {item.type === 'folder' ? (
                    <Folder className="h-12 w-12 text-blue-500" />
                  ) : (
                    <FileIcon type={item.fileType} />
                  )}
                  <div className="space-y-1 w-full">
                    <div className="font-medium text-sm truncate w-full" title={item.name}>
                      {item.name}
                    </div>
                    {item.type === 'file' && (
                      <div className="text-xs text-muted-foreground">
                        {item.size} • {item.date}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
            {(!currentFolder.children || currentFolder.children.length === 0) && (
              <div className="col-span-full text-center py-12 text-muted-foreground">
                This folder is empty
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
