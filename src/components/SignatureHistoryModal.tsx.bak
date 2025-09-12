import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "./ui/button";
import { X, Upload, History, Download } from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { uploadSignature, getStudentSignatures, getPrimarySignature, compareSignatures } from "@/lib/supabaseService";
import { supabase } from "@/lib/supabase";
import type { Signature } from "@/types";

interface SignatureHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  studentId: number;
  studentName: string;
  onSignatureSelect?: (signature: Signature) => void;
}

export function SignatureHistoryModal({ 
  isOpen, 
  onClose, 
  studentId, 
  studentName, 
  onSignatureSelect 
}: SignatureHistoryModalProps) {
  const [signatures, setSignatures] = useState<Signature[]>([]);
  const [primarySignature, setPrimarySignature] = useState<Signature | null>(null);
  const [isSettingPrimary, setIsSettingPrimary] = useState<Record<number, boolean>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load signatures when modal opens
  const loadSignatures = useCallback(async () => {
    try {
      setIsLoading(true);
      const [sigs, primary] = await Promise.all([
        getStudentSignatures(studentId),
        getPrimarySignature(studentId)
      ]);
      
      setSignatures(sigs);
      setPrimarySignature(primary);
    } catch (error) {
      console.error('Error loading signatures:', error);
      toast.error('Failed to load signatures');
    } finally {
      setIsLoading(false);
    }
  }, [studentId]);

  // Load signatures when modal opens or studentId changes
  useEffect(() => {
    if (isOpen) {
      loadSignatures();
    }
  }, [isOpen, loadSignatures]);

  // Handle file upload button click
  const handleUploadClick = useCallback((): void => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, []);

  // Set a signature as primary
  const handleSetPrimary = useCallback(async (signature: Signature) => {
    try {
      setIsSettingPrimary(prev => ({ ...prev, [signature.id]: true }));
      
      // Update the primary signature in the database
      await supabase
        .from('students')
        .update({ primary_signature_id: signature.id })
        .eq('id', studentId);
      
      // Update local state
      setPrimarySignature(signature);
      toast.success('Primary signature updated');
    } catch (error) {
      console.error('Error setting primary signature:', error);
      toast.error('Failed to set primary signature');
    } finally {
      setIsSettingPrimary(prev => ({ ...prev, [signature.id]: false }));
    }
  }, [studentId]);

  // Compare two signatures
  const handleCompareSignatures = useCallback(async (signature1: Signature, signature2: Signature) => {
    try {
      const score = await compareSignatures(signature1.id, signature2.id);
      toast.info(`Signature match score: ${(score * 100).toFixed(1)}%`, {
        duration: 5000,
      });
      return score;
    } catch (error) {
      console.error('Error comparing signatures:', error);
      toast.error('Failed to compare signatures');
      return 0;
    }
  }, []);

  // Handle signature selection
  const handleSignatureSelect = useCallback((signature: Signature) => {
    if (onSignatureSelect) {
      onSignatureSelect(signature);
      onClose();
    }
  }, [onSignatureSelect, onClose]);

  // Handle file selection and upload
  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    if (!studentId) return;
    
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/svg+xml'];
    if (!validTypes.includes(file.type)) {
      toast.error('Invalid file type. Please upload a PNG, JPG, or SVG file.');
      return;
    }
    
    // Validate file size (max 2MB)
    const maxSize = 2 * 1024 * 1024; // 2MB
    if (file.size > maxSize) {
      toast.error('File is too large. Maximum size is 2MB.');
      return;
    }

    try {
      setIsUploading(true);
      const { signature: newSignature } = await uploadSignature(studentId, file, {
        deviceInfo: {
          userAgent: navigator.userAgent,
          screenResolution: `${window.screen.width}x${window.screen.height}`,
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        },
      });

      // Add the new signature to the list
      setSignatures(prev => [newSignature, ...prev]);
      
      toast.success('Signature uploaded successfully');
    } catch (error) {
      console.error('Error uploading signature:', error);
      toast.error('Failed to upload signature');
    } finally {
      setIsUploading(false);
    }
  }, [studentId]);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => {
      if (!open) {
        // Prevent overlay/escape close globally; explicit close only
        onClose()
      }
    }}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex justify-between items-center">
            <DialogTitle>Signature History for {studentName}</DialogTitle>
            <DialogDescription>
              View and manage {studentName}'s signature history. Click on a signature to select it.
            </DialogDescription>
          </div>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={onClose}
            className="h-8 w-8"
          >
            <X className="h-4 w-4" />
          </Button>
        </DialogHeader>
        
        <div className="mt-4">
          <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex justify-between items-center">
              <h3 className="font-medium">Upload New Signature</h3>
              <div className="relative">
                <Button 
                  variant="outline"
                  disabled={isUploading}
                  className={isUploading ? 'opacity-50 cursor-not-allowed' : ''}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="mr-2 h-4 w-4" />
                  {isUploading ? 'Uploading...' : 'Select File'}
                </Button>
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="image/*"
                  className="hidden"
                  onChange={handleFileChange}
                  disabled={isUploading}
                />
              </div>
            </div>
            <p className="mt-2 text-sm text-gray-500">
              Supported formats: PNG, JPG, SVG. Max size: 2MB
            </p>
          </div>

          {isLoading ? (
            <div className="flex justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          ) : signatures.length === 0 ? (
            <div className="text-center py-12 border-2 border-dashed border-gray-200 rounded-lg">
              <p className="text-gray-500">No signatures found for this student.</p>
              <p className="text-sm text-gray-400 mt-1">Upload a signature to get started</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mt-4">
              {signatures.map((signature) => (
                <div 
                  key={signature.id} 
                  className={`relative group border rounded-md overflow-hidden ${
                    primarySignature?.id === signature.id ? 'ring-2 ring-primary' : ''
                  }`}
                >
                  <div 
                    className="cursor-pointer"
                    onClick={() => handleSignatureSelect(signature)}
                  >
                    <img
                      src={`${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/signatures/${signature.storage_path}`}
                      alt={`Signature ${signature.id}`}
                      className="w-full h-32 object-contain bg-white"
                      onError={(e) => {
                        // Fallback to a placeholder if the image fails to load
                        const target = e.target as HTMLImageElement;
                        target.src = '/signature-placeholder.png';
                      }}
                    />
                  </div>
                  <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center gap-2 p-2">
                    <div className="text-white text-xs text-center">
                      {new Date(signature.created_at).toLocaleDateString()}
                      {signature.quality_score && (
                        <div>Quality: {Math.round(signature.quality_score * 100)}%</div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className="text-white hover:bg-white/20"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSetPrimary(signature);
                        }}
                        disabled={isSettingPrimary[signature.id] || primarySignature?.id === signature.id}
                      >
                        {isSettingPrimary[signature.id] ? (
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        ) : (
                          <span className="text-xs">Primary</span>
                        )}
                      </Button>
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className="text-white hover:bg-white/20"
                        onClick={(e) => {
                          e.stopPropagation();
                          // Handle download
                          const link = document.createElement('a');
                          link.href = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/signatures/${signature.storage_path}`;
                          link.download = signature.file_name;
                          link.click();
                        }}
                      >
                        Download
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
