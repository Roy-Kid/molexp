import { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '../ui/dialog';
import { Button } from '../ui/button';
import { Label } from '../ui/label';
import { Input } from '../ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { useAppStore } from '@/store/useAppStore';
import { type Asset } from '@/types/workflow';
import { FileText, Database } from 'lucide-react';

interface NodeConfigDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (data: any) => void;
  nodeType: string | null;
  initialData?: any;
}

export const NodeConfigDialog = ({ isOpen, onClose, onConfirm, nodeType, initialData }: NodeConfigDialogProps) => {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const assets = useAppStore((state) => state.assets);

  useEffect(() => {
    if (isOpen) {
      if (initialData) {
        setFormData(initialData);
      } else if (nodeType) {
        setFormData({ label: getDefaultLabel(nodeType), ...getDefaultConfig(nodeType) });
      }
    }
  }, [isOpen, nodeType, initialData]);

  const getDefaultLabel = (type: string) => {
    switch (type) {
      case 'load-molecule': return 'Load Molecule';
      case 'optimize-geometry': return 'Optimize Geometry';
      case 'calc-energy': return 'Calculate Energy';
      case 'run-md': return 'Molecular Dynamics';
      case 'save-results': return 'Save Results';
      default: return 'Node';
    }
  };

  const getDefaultConfig = (type: string) => {
    switch (type) {
      case 'load-molecule': return { sourceType: 'file', value: '' };
      case 'optimize-geometry': return { method: 'DFT', basisSet: '6-31G*', maxIterations: 100 };
      case 'calc-energy': return { method: 'DFT', basisSet: '6-31G*' };
      case 'run-md': return { ensemble: 'NVT', temperature: 300, duration: 100, timeStep: 2 };
      case 'save-results': return { format: 'PDB', filename: 'output.pdb' };
      default: return {};
    }
  };

  const handleChange = (key: string, value: any) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  const renderFileSelector = () => {
    // Flatten assets for simple selection for now
    const getAllFiles = (nodes: Asset[]): Asset[] => {
      let files: Asset[] = [];
      nodes.forEach(node => {
        if (node.type === 'file') files.push(node);
        if (node.children) files = files.concat(getAllFiles(node.children));
      });
      return files;
    };
    const files = getAllFiles(assets);

    return (
      <Select value={formData.value} onValueChange={(val) => handleChange('value', val)}>
        <SelectTrigger>
          <SelectValue placeholder="Select a file..." />
        </SelectTrigger>
        <SelectContent>
          {files.map((file) => (
            <SelectItem key={file.id} value={file.name}>
              <div className="flex items-center">
                {file.fileType === 'pdb' || file.fileType === 'sdf' ? <Database className="mr-2 h-4 w-4" /> : <FileText className="mr-2 h-4 w-4" />}
                {file.name}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  };

  const renderFields = () => {
    switch (nodeType) {
      case 'load-molecule':
        return (
          <>
            <div className="grid gap-2">
              <Label>Source Type</Label>
              <Select value={formData.sourceType} onValueChange={(val) => handleChange('sourceType', val)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="file">File (Assets)</SelectItem>
                  <SelectItem value="smiles">SMILES String</SelectItem>
                  <SelectItem value="pdb_id">PDB ID</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label>Value</Label>
              {formData.sourceType === 'file' ? (
                renderFileSelector()
              ) : (
                <Input 
                  value={formData.value || ''} 
                  onChange={(e) => handleChange('value', e.target.value)} 
                  placeholder={formData.sourceType === 'smiles' ? 'C1=CC=CC=C1' : '1CRN'}
                />
              )}
            </div>
            {/* Molecule Preview Placeholder */}
            <div className="mt-4 p-4 border rounded bg-muted/20 flex items-center justify-center h-32">
              <div className="text-center text-muted-foreground">
                <Database className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <span>Molecule Structure Preview</span>
                {formData.value && <div className="text-xs mt-1 font-mono">{formData.value}</div>}
              </div>
            </div>
          </>
        );
      case 'optimize-geometry':
      case 'calc-energy':
        return (
          <>
            <div className="grid gap-2">
              <Label>Method</Label>
              <Select value={formData.method} onValueChange={(val) => handleChange('method', val)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="HF">Hartree-Fock</SelectItem>
                  <SelectItem value="DFT">DFT</SelectItem>
                  <SelectItem value="Semi-Empirical">Semi-Empirical</SelectItem>
                  {nodeType === 'calc-energy' && <SelectItem value="MP2">MP2</SelectItem>}
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label>Basis Set</Label>
              <Select value={formData.basisSet} onValueChange={(val) => handleChange('basisSet', val)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="STO-3G">STO-3G</SelectItem>
                  <SelectItem value="3-21G">3-21G</SelectItem>
                  <SelectItem value="6-31G*">6-31G*</SelectItem>
                  <SelectItem value="cc-pVDZ">cc-pVDZ</SelectItem>
                  {nodeType === 'calc-energy' && <SelectItem value="cc-pVTZ">cc-pVTZ</SelectItem>}
                </SelectContent>
              </Select>
            </div>
            {nodeType === 'optimize-geometry' && (
               <div className="grid gap-2">
                 <Label>Max Iterations</Label>
                 <Input type="number" value={formData.maxIterations || ''} onChange={(e) => handleChange('maxIterations', parseInt(e.target.value))} />
               </div>
            )}
          </>
        );
      case 'run-md':
        return (
          <>
            <div className="grid gap-2">
              <Label>Ensemble</Label>
              <Select value={formData.ensemble} onValueChange={(val) => handleChange('ensemble', val)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="NVE">NVE (Microcanonical)</SelectItem>
                  <SelectItem value="NVT">NVT (Canonical)</SelectItem>
                  <SelectItem value="NPT">NPT (Isothermal-Isobaric)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="grid gap-2">
                <Label>Temperature (K)</Label>
                <Input type="number" value={formData.temperature || ''} onChange={(e) => handleChange('temperature', parseFloat(e.target.value))} />
              </div>
              <div className="grid gap-2">
                <Label>Duration (ps)</Label>
                <Input type="number" value={formData.duration || ''} onChange={(e) => handleChange('duration', parseFloat(e.target.value))} />
              </div>
            </div>
          </>
        );
      case 'save-results':
        return (
          <div className="space-y-2">
            <Label htmlFor="filename">Filename</Label>
            <Input 
              id="filename" 
              placeholder="results.json" 
              value={formData.filename || ''} 
              onChange={e => handleChange('filename', e.target.value)} 
            />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Configure {formData.label}</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="label">Label</Label>
            <Input
              id="label"
              value={formData.label || ''}
              onChange={(e) => handleChange('label', e.target.value)}
            />
          </div>
          {renderFields()}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={() => onConfirm(formData)}>Confirm</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
