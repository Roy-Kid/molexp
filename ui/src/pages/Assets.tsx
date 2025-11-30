import { useState, useEffect } from 'react';
import { Folder, File, FileText, Database, Search, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';
import { useAppStore } from '@/store/useAppStore';
import { UploadAssetDialog } from '@/components/UploadAssetDialog';
import { DetailPanel } from '@/components/DetailPanel';

const FileIcon = ({ type }: { type: string }) => {
  switch (type?.toLowerCase()) {
    case 'pdb':
    case 'sdf':
    case 'mol2':
      return <Database className="h-8 w-8 text-blue-500" />;
    case 'log':
    case 'txt':
    case 'out':
      return <FileText className="h-8 w-8 text-gray-500" />;
    case 'json':
    case 'yaml':
    case 'yml':
      return <File className="h-8 w-8 text-yellow-500" />;
    default:
      return <File className="h-8 w-8 text-gray-400" />;
  }
};

export const Assets = () => {
  const assets = useAppStore((state) => state.assets);
  const fetchAssets = useAppStore((state) => state.fetchAssets);
  const isLoading = useAppStore((state) => state.isLoading);
  
  const [selectedAsset, setSelectedAsset] = useState<any>(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchAssets();
  }, [fetchAssets]);

  const filteredAssets = assets.filter((asset: any) => {
    const name = asset.metadata?.original_filename || asset.asset_id;
    return name.toLowerCase().includes(searchTerm.toLowerCase()) || 
           asset.asset_id.toLowerCase().includes(searchTerm.toLowerCase());
  });

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center text-muted-foreground">Loading assets...</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-6 py-4 border-b">
        <h2 className="text-2xl font-bold tracking-tight">Assets</h2>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input 
              placeholder="Search assets..." 
              className="pl-8 w-64" 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <UploadAssetDialog />
        </div>
      </div>
      
      <div className="flex-1 flex overflow-hidden">
        {/* Main Content - Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="mb-4 text-sm text-muted-foreground">
            All Assets ({filteredAssets.length})
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {filteredAssets.map((asset: any) => (
              <Card 
                key={asset.asset_id} 
                className={`cursor-pointer hover:shadow-md transition-shadow ${selectedAsset?.asset_id === asset.asset_id ? 'ring-2 ring-primary' : ''}`}
                onClick={() => setSelectedAsset(asset)}
              >
                <CardContent className="p-4 flex flex-col items-center text-center gap-3">
                  <FileIcon type={asset.format} />
                  <div className="space-y-1 w-full">
                    <div className="font-medium text-sm truncate w-full" title={asset.metadata?.original_filename || asset.asset_id}>
                      {asset.metadata?.original_filename || asset.asset_id}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {(asset.size / 1024).toFixed(1)} KB • {new Date(asset.created).toLocaleDateString()}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
            {filteredAssets.length === 0 && (
              <div className="col-span-full text-center py-12 text-muted-foreground">
                No assets found
              </div>
            )}
          </div>
        </div>

        {/* Right Sidebar - Details */}
        {selectedAsset && (
          <div className="w-80 border-l bg-background overflow-y-auto">
            <div className="p-4 flex justify-between items-center border-b">
              <h3 className="font-semibold">Asset Details</h3>
              <Button variant="ghost" size="sm" onClick={() => setSelectedAsset(null)}>Close</Button>
            </div>
            <DetailPanel nodeId={selectedAsset.asset_id} nodeType="asset" />
          </div>
        )}
      </div>
    </div>
  );
};
