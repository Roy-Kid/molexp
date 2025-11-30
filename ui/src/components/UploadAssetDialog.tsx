import React, { useState, useRef } from 'react';
import { Upload, File, Loader, CheckCircle, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { assetApi } from '@/services/api';
import { useAppStore } from '@/store/useAppStore';

export const UploadAssetDialog = () => {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const fetchAssets = useAppStore((state) => state.fetchAssets);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setSuccess(false);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      // Use assetApi from services which uses axios (easier for progress if we added it, 
      // but standard axios post doesn't expose progress callback easily in the wrapper)
      // For now, we'll just await it.
      await assetApi.upload(file);
      
      setSuccess(true);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      
      // Refresh assets list
      fetchAssets();
      
      // Close dialog after a delay
      setTimeout(() => {
        setOpen(false);
        setSuccess(false);
      }, 1500);
      
    } catch (err) {
      console.error('Upload failed:', err);
      setError('Failed to upload asset. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
      setSuccess(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="gap-2">
          <Upload className="h-4 w-4" />
          Upload Asset
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Upload Asset</DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          <div 
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              file ? 'border-primary/50 bg-primary/5' : 'border-muted-foreground/25 hover:border-primary/50'
            }`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {file ? (
              <div className="flex flex-col items-center gap-2">
                <File className="h-10 w-10 text-primary" />
                <div className="font-medium text-sm truncate max-w-[200px]">{file.name}</div>
                <div className="text-xs text-muted-foreground">
                  {(file.size / 1024).toFixed(1)} KB
                </div>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="mt-2 h-6 text-xs text-red-500 hover:text-red-600 hover:bg-red-50"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                >
                  Remove
                </Button>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                <Upload className="h-10 w-10 text-muted-foreground/50" />
                <div className="text-sm font-medium text-muted-foreground">
                  Drag & drop or click to select
                </div>
                <div className="text-xs text-muted-foreground/50">
                  Any file type supported
                </div>
              </div>
            )}
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              onChange={handleFileChange} 
            />
          </div>

          {error && (
            <div className="flex items-center gap-2 text-sm text-red-500 bg-red-50 p-2 rounded">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          {success && (
            <div className="flex items-center gap-2 text-sm text-green-500 bg-green-50 p-2 rounded">
              <CheckCircle className="h-4 w-4" />
              Upload successful!
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2">
          <Button variant="outline" onClick={() => setOpen(false)} disabled={uploading}>
            Cancel
          </Button>
          <Button onClick={handleUpload} disabled={!file || uploading || success}>
            {uploading ? (
              <>
                <Loader className="mr-2 h-4 w-4 animate-spin" />
                Uploading...
              </>
            ) : (
              'Upload'
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
