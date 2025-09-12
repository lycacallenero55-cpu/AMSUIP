import React, { useEffect, useRef, useState } from 'react';
import Layout from '@/components/Layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useToast } from '@/components/ui/use-toast';
import { UnsavedChangesDialog } from '@/components/UnsavedChangesDialog';
import { useUnsavedChanges } from '@/hooks/useUnsavedChanges';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Upload, 
  Camera, 
  Brain, 
  Scan, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Loader2,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  User,
  Trash2,
  AlertTriangle,
  X,
  MoreVertical
} from 'lucide-react';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu';
import { Switch } from '@/components/ui/switch';
import { aiService, AI_CONFIG } from '@/lib/aiService';
import { fetchStudents } from '@/lib/supabaseService';
import type { Student, StudentTrainingCard as StudentTrainingCardType, TrainingFile } from '@/types';
import { Progress } from '@/components/ui/progress';
import StudentTrainingCard from '@/components/StudentTrainingCard';

type TrainedModel = {
  id: string | number;
  student_name?: string;
  student?: { id?: number; student_id?: string; firstname?: string; surname?: string; full_name?: string };
  student_full_name?: string;
  model_path?: string;
  model?: { path?: string };
  artifact_path?: string;
  training_date?: string;
  created_at?: string;
  accuracy?: number;
};

const SignatureAI = () => {
  const { toast } = useToast();
  
  // Multi-Student Training State
  const [studentCards, setStudentCards] = useState<StudentTrainingCardType[]>([
    {
      id: '1',
      student: null,
      genuineFiles: [],
      forgedFiles: [],
      isExpanded: true
    }
  ]);
  const [isTraining, setIsTraining] = useState(false);
  const isLocked = isTraining;
  const [trainingResult, setTrainingResult] = useState<{
    success: boolean;
    message: string;
    accuracy?: number;
    val_accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    train_time_s?: number;
    profile?: {
      status: string;
      num_samples: number;
      last_trained_at?: string;
    };
    calibration?: {
      threshold: number;
      far: number;
      frr: number;
    };
  } | null>(null);
  
  // Training Progress State
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState<string>('');
  const [trainingStage, setTrainingStage] = useState<'idle' | 'preprocessing' | 'training' | 'validation' | 'completed' | 'error'>('idle');
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState<string>('');
  const [jobId, setJobId] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const trainingStartTimeRef = useRef<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  
  // Real-time training logs state
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [currentEpochProgress, setCurrentEpochProgress] = useState<{
    epoch: number;
    totalEpochs: number;
    batch: number;
    totalBatches: number;
    accuracy: number;
    loss: number;
    valAccuracy: number;
    valLoss: number;
  } | null>(null);
  
  // Verification Section State
  const [verificationFile, setVerificationFile] = useState<File | null>(null);
  const [verificationPreview, setVerificationPreview] = useState<string>('');
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationResult, setVerificationResult] = useState<{
    success: boolean;
    match: boolean;
    score: number;
    message?: string;
    predicted_student_id?: number | null;
    predicted_student?: {
      id: number;
      student_id: string;
      firstname: string;
      surname: string;
    };
  } | null>(null);
  const [useCamera, setUseCamera] = useState(false);
  
  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalImageIndex, setModalImageIndex] = useState(0);
  const [modalImages, setModalImages] = useState<string[]>([]);
  
  // Student Selection State
  const [isStudentDialogOpen, setIsStudentDialogOpen] = useState(false);
  const [studentSearch, setStudentSearch] = useState('');
  const [debouncedStudentSearch, setDebouncedStudentSearch] = useState('');
  const [isStudentSearching, setIsStudentSearching] = useState(false);
  const [allStudents, setAllStudents] = useState<Student[]>([]);
  const [isLoadingStudents, setIsLoadingStudents] = useState(false);
  const [currentCardId, setCurrentCardId] = useState<string>('');
  const [studentSelectMode, setStudentSelectMode] = useState<'assign' | 'bulkAdd'>('assign');
  const [selectedStudentIds, setSelectedStudentIds] = useState<Set<number>>(new Set());
  const [studentPage, setStudentPage] = useState<number>(1);
  const STUDENTS_PER_PAGE = 100;
  const [showSelectedStudents, setShowSelectedStudents] = useState<boolean>(false);
  
  // Student Form Dialog State
  const [isStudentFormDialogOpen, setIsStudentFormDialogOpen] = useState(false);
  const [currentFormCardId, setCurrentFormCardId] = useState<string | null>(null);
  const [trainingImagesSet, setTrainingImagesSet] = useState<'genuine' | 'forged'>('genuine');
  
  // Toggle between Student Cards and Trained Models view
  const [isViewingModels, setIsViewingModels] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);
  const [confirmDeleteModelId, setConfirmDeleteModelId] = useState<string | number | null>(null);
  
  // Date-based model view state
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [dateGroupedModels, setDateGroupedModels] = useState<Record<string, { global?: TrainedModel; individual: TrainedModel[] }>>({});
  
  // Training mode (hybrid by design; no label shown)
  const [useGPU, setUseGPU] = useState(true);

  // Generate mock models with hybrid training data
  const generateMockHybridModels = React.useCallback((): Record<string, { global?: TrainedModel; individual: TrainedModel[] }> => {
    const now = new Date();
    const grouped: Record<string, { global?: TrainedModel; individual: TrainedModel[] }> = {};
    
    // Generate models for the last 5 days
    for (let dayOffset = 0; dayOffset < 5; dayOffset++) {
      const d = new Date(now.getTime() - dayOffset * 86400000);
      const yyyy = d.getFullYear();
      const mm = String(d.getMonth() + 1).padStart(2, '0');
      const dd = String(d.getDate()).padStart(2, '0');
      const dateKey = `${yyyy}-${mm}-${dd}`;
      const dateIso = `${dateKey}T12:00:00Z`;
      
      // Add global model for this date (every other day)
      if (dayOffset % 2 === 0) {
        grouped[dateKey] = {
          global: {
            id: `global-${dateKey}`,
            student_name: 'Global Model',
            student_full_name: 'Global Model',
            model_path: `/models/global_model_${dateKey}.keras`,
            training_date: dateIso,
            created_at: dateIso,
            accuracy: Math.round((85 + Math.random() * 15) * 10) / 1000,
          } as TrainedModel,
          individual: []
        };
      } else {
        grouped[dateKey] = { individual: [] };
      }
      
      // Add 3-6 individual models per date
      const individualCount = 3 + Math.floor(Math.random() * 4);
      for (let i = 0; i < individualCount; i++) {
        const acc = Math.round((80 + Math.random() * 20) * 10) / 1000;
        grouped[dateKey].individual.push({
          id: `individual-${dateKey}-${i + 1}`,
          student_name: `Student ${dayOffset * 10 + i + 1}`,
          student_full_name: `Student ${dayOffset * 10 + i + 1}`,
          model_path: `/models/individual_model_${dateKey}_${i + 1}.keras`,
          training_date: dateIso,
          created_at: dateIso,
          accuracy: acc,
        } as TrainedModel);
      }
    }
    
    return grouped;
  }, []);

  // Legacy mock models for backward compatibility
  const generateMockModels = React.useCallback((count: number = 20): TrainedModel[] => {
    const now = new Date();
    return Array.from({ length: count }).map((_, i) => {
      const d = new Date(now.getTime() - i * 86400000);
      const yyyy = d.getFullYear();
      const mm = String(d.getMonth() + 1).padStart(2, '0');
      const dd = String(d.getDate()).padStart(2, '0');
      const dateIso = `${yyyy}-${mm}-${dd}T12:00:00Z`;
      const acc = Math.round((80 + Math.random() * 20) * 10) / 1000; // 0.800 - 0.999
      return {
        id: `mock-${i + 1}`,
        student_name: `Student ${i + 1}`,
        student_full_name: `Student ${i + 1}`,
        model_path: `/models/model_${i + 1}.keras`,
        training_date: dateIso,
        created_at: dateIso,
        accuracy: acc,
      } as TrainedModel;
    });
  }, []);

  // Unsaved changes handling
  const navigate = useNavigate();
  
  // Page-scoped set of uploaded image hashes to prevent duplicates within this page
  const [uploadedImageHashes, setUploadedImageHashes] = useState<Set<string>>(new Set());
  
  // Toggle for duplicate image prevention
  const [preventDuplicates, setPreventDuplicates] = useState<boolean>(true);
  
  // Function to handle toggle change
  const handlePreventDuplicatesToggle = (checked: boolean) => {
    setPreventDuplicates(checked);
    // If turning off prevention, clear all hashes to allow re-uploading
    if (!checked) {
      setUploadedImageHashes(new Set());
      toast({
        title: 'Duplicate prevention disabled',
        description: 'You can now re-upload the same images.',
      });
    } else {
      toast({
        title: 'Duplicate prevention enabled',
        description: 'Duplicate images will be blocked from upload.',
      });
    }
  };

  // Compute a stable content hash for a File using SHA-256
  const computeFileHash = async (file: File): Promise<string> => {
    const buffer = await file.arrayBuffer();
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(digest));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
    return hashHex;
  };
  const location = useLocation();
  const [pendingNavigation, setPendingNavigation] = useState<string | null>(null);
  
  const {
    hasUnsavedChanges,
    showConfirmDialog,
    markAsChanged,
    markAsSaved,
    handleClose,
    confirmClose,
    cancelClose,
    handleOpenChange,
  } = useUnsavedChanges({
    onClose: () => {
      if (pendingNavigation) {
        navigate(pendingNavigation);
        setPendingNavigation(null);
      }
    },
    enabled: true,
  });

  const markDirty = React.useCallback(() => markAsChanged(), [markAsChanged]);

  // Intercept navigation attempts
  React.useEffect(() => {
    const handleClick = (event: Event) => {
      const target = event.target as HTMLElement;
      const link = target.closest('a[href]') as HTMLAnchorElement;
      
      if (link && hasUnsavedChanges) {
        const href = link.getAttribute('href');
        if (href && href.startsWith('/') && href !== location.pathname) {
          event.preventDefault();
          setPendingNavigation(href);
          handleClose();
        }
      }
    };

    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, [hasUnsavedChanges, location.pathname, handleClose]);

  // Warn before unload if there are any interactions/changes
  React.useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (!hasUnsavedChanges) return;
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [hasUnsavedChanges]);

  // Fetch students on component mount
  React.useEffect(() => {
    const loadStudents = async () => {
      setIsLoadingStudents(true);
      try {
        const students = await fetchStudents();
        const sortedStudents = students.sort((a, b) => {
          const nameA = `${a.firstname} ${a.surname}`.toLowerCase();
          const nameB = `${b.firstname} ${b.surname}`.toLowerCase();
          return nameA.localeCompare(nameB);
        });
        setAllStudents(sortedStudents);
      } catch (error) {
        console.error('Error loading students:', error);
        toast({
          title: "Error",
          description: "Failed to load students",
          variant: "destructive",
        });
      } finally {
        setIsLoadingStudents(false);
      }
    };
    loadStudents();
  }, [toast]);

  React.useEffect(() => {
    setIsStudentSearching(true);
    const t = setTimeout(() => {
      setDebouncedStudentSearch(studentSearch.trim());
      setIsStudentSearching(false);
    }, 300);
    return () => clearTimeout(t);
  }, [studentSearch]);

  const baseFilteredStudents = debouncedStudentSearch 
    ? allStudents.filter((s) => (
        s.student_id.includes(debouncedStudentSearch) ||
        `${s.firstname} ${s.surname}`.toLowerCase().includes(debouncedStudentSearch.toLowerCase())
      ))
    : allStudents;

  const selectedElsewhereIds = React.useMemo(() => {
    const ids = new Set<number>();
    studentCards.forEach(card => {
      if (card.id !== currentCardId && card.student) ids.add(card.student.id);
    });
    return ids;
  }, [studentCards, currentCardId]);

  const visibleStudents = showSelectedStudents
    ? baseFilteredStudents
    : baseFilteredStudents.filter((s) => !selectedElsewhereIds.has(s.id));

  const totalStudentPages = Math.max(1, Math.ceil(visibleStudents.length / STUDENTS_PER_PAGE));
  const pagedStudents = visibleStudents.slice(
    (studentPage - 1) * STUDENTS_PER_PAGE,
    studentPage * STUDENTS_PER_PAGE
  );

  // Per-page selection helpers (exclude students already selected elsewhere)
  const selectablePagedStudentIds = React.useMemo(() => {
    return new Set<number>(
      pagedStudents
        .filter((s) => !selectedElsewhereIds.has(s.id))
        .map((s) => s.id)
    );
  }, [pagedStudents, selectedElsewhereIds]);

  const selectedCountOnPage = React.useMemo(() => {
    let count = 0;
    selectablePagedStudentIds.forEach((id) => {
      if (selectedStudentIds.has(id)) count++;
    });
    return count;
  }, [selectablePagedStudentIds, selectedStudentIds]);

  const allSelectedOnPage = selectablePagedStudentIds.size > 0 && selectedCountOnPage === selectablePagedStudentIds.size;

  const verificationInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Refs for components
  const verificationCardRef = useRef<HTMLDivElement>(null);
  const studentContainerRef = useRef<HTMLDivElement>(null);
  

  // Validation functions
  const hasUploadedImages = (card: StudentTrainingCardType) => {
    return card.genuineFiles.length > 0 || card.forgedFiles.length > 0;
  };

  const canTrainModel = () => {
    return studentCards.some(card => card.student !== null && hasUploadedImages(card));
  };

  const getTotalTrainingData = () => {
    return studentCards.reduce((acc, card) => ({
      genuine: acc.genuine + card.genuineFiles.length,
      forged: acc.forged + card.forgedFiles.length
    }), { genuine: 0, forged: 0 });
  };

  const getSelectedStudentIds = () => {
    return studentCards
      .filter(card => card.student !== null)
      .map(card => card.student!.id);
  };

  const isStudentAlreadySelected = (studentId: number, currentCardId: string) => {
    return studentCards.some(card => 
      card.id !== currentCardId && 
      card.student !== null && 
      card.student.id === studentId
    );
  };

  const getTrainModelErrorMessage = () => {
    const hasAnyData = studentCards.some(card => hasUploadedImages(card));
    const hasAnyStudent = studentCards.some(card => card.student !== null);
    
    if (!hasAnyStudent && !hasAnyData) {
      return "Please add at least one student and upload signature images to train the model.";
    }
    if (!hasAnyStudent) {
      return "Please select students for the training cards.";
    }
    if (!hasAnyData) {
      return "Please upload at least one signature image to train the model.";
    }
    return "";
  };

  // Student Card Management Functions
  const addStudentCard = React.useCallback(() => {
    if (isLocked) return;
    // Deprecated: no longer create empty cards.
    // Open the student selector in bulk add mode instead
    setStudentSelectMode('bulkAdd');
    setSelectedStudentIds(new Set());
    setStudentPage(1);
    setIsStudentDialogOpen(true);
  }, [isLocked]);

  const removeStudentCard = React.useCallback((cardId: string) => {
    setStudentCards(prev => {
      if (prev.length <= 1) {
        // Reset the only card instead of removing it
        return prev.map(card => {
          if (card.id !== cardId) return card;
          // Revoke previews to avoid memory leaks
          card.genuineFiles.forEach(file => URL.revokeObjectURL(file.preview));
          card.forgedFiles.forEach(file => URL.revokeObjectURL(file.preview));
          return {
            ...card,
            student: null,
            genuineFiles: [],
            forgedFiles: [],
          };
        });
      }
      
      const card = prev.find(c => c.id === cardId);
      if (card) {
        card.genuineFiles.forEach(file => URL.revokeObjectURL(file.preview));
        card.forgedFiles.forEach(file => URL.revokeObjectURL(file.preview));
      }
      
      return prev.filter(c => c.id !== cardId);
    });
  }, [toast]);

  const updateStudentCard = React.useCallback((cardId: string, updates: Partial<StudentTrainingCardType>) => {
    setStudentCards(prev => prev.map(card => 
      card.id === cardId ? { ...card, ...updates } : card
    ));
  }, []);

  const handleStudentSelection = (student: Student, keepFormOpen = false) => {
    if (currentCardId) {
      // Check if student is already selected in another card
      const isAlreadySelected = isStudentAlreadySelected(student.id, currentCardId);
      
      if (isAlreadySelected) {
        toast({
          title: "Student Already Selected",
          description: "This student is already selected in another training card. Please choose a different student.",
          variant: "destructive"
        });
        return;
      }

      const card = studentCards.find(c => c.id === currentCardId);
      if (card && hasUploadedImages(card) && card.student && card.student.id !== student.id) {
        card.genuineFiles.forEach(file => URL.revokeObjectURL(file.preview));
        card.forgedFiles.forEach(file => URL.revokeObjectURL(file.preview));
        updateStudentCard(currentCardId, {
          student,
          genuineFiles: [],
          forgedFiles: []
        });
        toast({
          title: "Student Changed",
          description: "All uploaded images have been cleared for the new student.",
        });
      } else {
        updateStudentCard(currentCardId, { student });
        // Load persisted signatures for preview from S3
        (async () => {
          try {
            const persisted = await aiService.listSignatures(student.id);
            const toPreview = (rec: { id:number; s3_url:string; s3_key:string; label:'genuine'|'forged' }) => ({ file: new File([], rec.s3_url), preview: rec.s3_url, id: rec.id, s3Key: rec.s3_key, label: rec.label } as TrainingFile);
            updateStudentCard(currentCardId, {
              genuineFiles: persisted.filter(x => x.label === 'genuine').map(x => toPreview(x)),
              forgedFiles: persisted.filter(x => x.label === 'forged').map(x => toPreview(x)),
            });
          } catch (e) {
            // ignore
          }
        })();
        toast({
          title: "Student Selected",
          description: `${student.firstname} ${student.surname} has been added to the training card.`,
        });
      }
    }
    
    // Only close the student dialog if not keeping form open
    if (!keepFormOpen) {
      setIsStudentDialogOpen(false);
      setCurrentCardId('');
    }
    markDirty();
  };

  const openStudentDialog = (cardId: string) => {
    setCurrentCardId(cardId);
    setStudentSelectMode('assign');
    setSelectedStudentIds(new Set());
    setStudentPage(1);
    setIsStudentDialogOpen(true);
  };

  const openStudentFormDialog = (cardId: string) => {
    setCurrentFormCardId(cardId);
    setIsStudentFormDialogOpen(true);
    setTrainingImagesSet('genuine');
  };

  // Training Functions
  const validateFiles = (files: File[]): File[] => {
    const MAX_SIZE_BYTES = 10 * 1024 * 1024;
    const valid: File[] = [];
    let rejected = 0;
    files.forEach((f) => {
      if (!f.type.startsWith('image/')) {
        rejected++;
        return;
      }
      if (f.size > MAX_SIZE_BYTES) {
        rejected++;
        return;
      }
      valid.push(f);
    });
    if (rejected > 0) {
      toast({
        title: 'Some files were ignored',
        description: 'Only image files up to 10MB are allowed.',
        variant: 'destructive',
      });
    }
    return valid;
  };

  const handleTrainingFilesChange = async (files: File[], setType: 'genuine' | 'forged', cardId: string) => {
    const safeFiles = validateFiles(files);
    if (safeFiles.length === 0) return;

    // Compute hashes and filter duplicates within this page (if enabled)
    const fileEntries = await Promise.all(safeFiles.map(async (file) => {
      const hash = await computeFileHash(file);
      return { file, hash };
    }));

    const uniqueEntries = preventDuplicates 
      ? fileEntries.filter(({ hash }) => !uploadedImageHashes.has(hash))
      : fileEntries;

    if (preventDuplicates && uniqueEntries.length === 0) {
      toast({
        title: 'Duplicate images blocked',
        description: 'This image has already been uploaded on this page.',
        variant: 'destructive',
      });
      return;
    }

    // Warn if some were filtered out (only when prevention is enabled)
    if (preventDuplicates && uniqueEntries.length < fileEntries.length) {
      toast({
        title: 'Some duplicates were ignored',
        description: 'This image has already been uploaded on this page.',
        variant: 'destructive',
      });
    }

    // Upload to backend → S3 and use returned URLs for persistent previews
    const card = studentCards.find(c => c.id === cardId);
    if (!card?.student) return;
    const uploaded: TrainingFile[] = [];
    for (const { file } of uniqueEntries) {
      try {
        const rec = await aiService.uploadSignature(card.student.id, setType, file);
        uploaded.push({ file: new File([], rec.s3_url), preview: rec.s3_url, id: rec.id, s3Key: rec.s3_key, label: rec.label });
      } catch (e: any) {
        // Duplicate handling (HTTP 409)
        if (e instanceof Error && e.message.includes('Duplicate')) {
          toast({ title: 'Duplicate image', description: 'This image was already uploaded.', variant: 'destructive' });
          continue;
        }
        // fallback to local preview if upload fails
        try {
          const fallback = file.type.startsWith('image/') && file.type !== 'image/tiff' && !file.name.toLowerCase().endsWith('.tif') && !file.name.toLowerCase().endsWith('.tiff')
            ? URL.createObjectURL(file)
            : await aiService.getPreviewURL(file);
          uploaded.push({ file, preview: fallback });
        } catch {
          // ignore
        }
      }
    }

    // Add to page-level hash set (only when prevention is enabled)
    if (preventDuplicates) {
      setUploadedImageHashes(prev => {
        const next = new Set(prev);
        uniqueEntries.forEach(({ hash }) => next.add(hash));
        return next;
      });
    }

    // Update the target card's files
    setStudentCards(prev => prev.map(card => {
      if (card.id === cardId) {
        if (setType === 'genuine') {
          return { ...card, genuineFiles: [...card.genuineFiles, ...uploaded] };
        } else {
          return { ...card, forgedFiles: [...card.forgedFiles, ...uploaded] };
        }
      }
      return card;
    }));
    markDirty();
  };

  // (Deletion remains as previously implemented below in file)

  const removeTrainingFile = (index: number, setType: 'genuine' | 'forged', cardId: string) => {
    setStudentCards(prev => prev.map(card => {
      if (card.id === cardId) {
        const list = setType === 'genuine' ? [...card.genuineFiles] : [...card.forgedFiles];
        const removed = list[index];
        // Best-effort backend delete when we know record id
        if (removed?.id) {
          aiService.deleteSignature(removed.id, removed.s3Key).catch(() => { /* ignore */ });
        }
        // Revoke only blob URLs
        if (removed && removed.preview.startsWith('blob:')) {
          URL.revokeObjectURL(removed.preview);
        }
        list.splice(index, 1);
        // Remove hash from page-level set (only when prevention is enabled)
        if (preventDuplicates) {
          removed?.file.arrayBuffer().then((buf) => crypto.subtle.digest('SHA-256', buf)).then((digest) => {
            const hashArray = Array.from(new Uint8Array(digest));
            const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
            setUploadedImageHashes(prev => { const next = new Set(prev); next.delete(hashHex); return next; });
          }).catch(() => { /* noop */ });
        }
        return setType === 'genuine' ? { ...card, genuineFiles: list } : { ...card, forgedFiles: list };
      }
      return card;
    }));
    markDirty();
  };

  const removeAllSamples = React.useCallback((cardId: string) => {
    setStudentCards(prev => prev.map(card => {
      if (card.id === cardId) {
        // Revoke all object URLs to prevent memory leaks
        card.genuineFiles.forEach(file => URL.revokeObjectURL(file.preview));
        card.forgedFiles.forEach(file => URL.revokeObjectURL(file.preview));
        // Remove hashes for these files from the page-level set (only when prevention is enabled)
        if (preventDuplicates) {
          const removeHashes = async () => {
            const all = [...card.genuineFiles, ...card.forgedFiles];
            for (const tf of all) {
              try {
                const buf = await tf.file.arrayBuffer();
                const digest = await crypto.subtle.digest('SHA-256', buf);
                const hashArray = Array.from(new Uint8Array(digest));
                const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
                setUploadedImageHashes(prev => {
                  const next = new Set(prev);
                  next.delete(hashHex);
                  return next;
                });
              } catch {
                // ignore
              }
            }
          };
          removeHashes();
        }
        return { ...card, genuineFiles: [], forgedFiles: [] };
      }
      return card;
    }));
    markDirty();
  }, [markDirty]);

  const handleTrainModel = async () => {
    if (!canTrainModel()) {
      toast({
        title: "Cannot Train Model",
        description: getTrainModelErrorMessage(),
        variant: "destructive",
      });
      return;
    }

    setIsTraining(true);
    setTrainingResult(null);
    setTrainingProgress(0);
    setTrainingStage('preprocessing');
    setEstimatedTimeRemaining('');
    setElapsedMs(0);
    setTrainingLogs([]);
    setCurrentEpochProgress(null);
    trainingStartTimeRef.current = Date.now();
    
    setTrainingLogs(['Training started for multiple students...']);
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    try {
      const allGenuineFiles: File[] = [];
      const allForgedFiles: File[] = [];
      const studentIds: string[] = [];
      
      studentCards.forEach(card => {
        if (card.student && hasUploadedImages(card)) {
          studentIds.push(card.student.student_id);
          allGenuineFiles.push(...card.genuineFiles.map(f => f.file));
          allForgedFiles.push(...card.forgedFiles.map(f => f.file));
        }
      });

      const asyncResponse = useGPU
        ? await aiService.startGPUTraining(
            studentIds.join(','),
            allGenuineFiles,
            allForgedFiles,
            true
          )
        : await aiService.startAsyncTraining(
            studentIds.join(','),
            allGenuineFiles,
            allForgedFiles,
            'hybrid'
          );
      
      setJobId(asyncResponse.job_id);
      setTrainingLogs(prev => [...prev, `Job created: ${asyncResponse.job_id}`]);
      
      const eventSource = aiService.subscribeToJobProgress(
        asyncResponse.job_id,
        (job) => {
          if (job.current_stage) setTrainingStage(job.current_stage as 'idle' | 'preprocessing' | 'training' | 'validation' | 'completed' | 'error');
          setTrainingStatus(job.current_stage || '');
          
          if (job.current_stage && job.current_stage !== 'idle') {
            setTrainingLogs(prev => {
              const newLogs = [...prev];
              const stageLog = `Stage: ${job.current_stage} - Progress: ${Math.round(job.progress || 0)}%`;
              if (newLogs.length === 0 || !newLogs[newLogs.length - 1].includes(job.current_stage)) {
                newLogs.push(stageLog);
              }
              return newLogs.slice(-15);
            });
          }
          
          const newProgress = Math.max(0, Math.min(100, job.progress || 0));
          setTrainingProgress((prev) => Math.max(prev, newProgress));
          
          if (typeof job.estimated_time_remaining === 'number') {
            const minutes = Math.floor(job.estimated_time_remaining / 60);
            const seconds = job.estimated_time_remaining % 60;
            setEstimatedTimeRemaining(`~${minutes}:${seconds.toString().padStart(2, '0')} remaining`);
          } else {
            setEstimatedTimeRemaining('');
          }
          
          if (job.training_metrics) {
            const metrics = job.training_metrics;
            if (metrics.current_epoch > 0) {
              let logEntry = `Epoch ${metrics.current_epoch}/${metrics.total_epochs}`;
              logEntry += ` - Accuracy: ${(metrics.accuracy * 100).toFixed(1)}% - Loss: ${metrics.loss.toFixed(4)}`;
              if (metrics.val_accuracy > 0) {
                logEntry += ` - Val Accuracy: ${(metrics.val_accuracy * 100).toFixed(1)}% - Val Loss: ${metrics.val_loss.toFixed(4)}`;
              }
              
              setTrainingLogs(prev => {
                const newLogs = [...prev];
                const lastLogIndex = newLogs.length - 1;
                const epochKey = `Epoch ${metrics.current_epoch}/${metrics.total_epochs}`;
                
                if (lastLogIndex >= 0 && newLogs[lastLogIndex].startsWith(epochKey)) {
                  newLogs[lastLogIndex] = logEntry;
                } else {
                  newLogs.push(logEntry);
                }
                return newLogs.slice(-15);
              });
              
              setCurrentEpochProgress({
                epoch: metrics.current_epoch || 0,
                totalEpochs: metrics.total_epochs || 0,
                batch: 0,
                totalBatches: 0,
                accuracy: metrics.accuracy || 0,
                loss: metrics.loss || 0,
                valAccuracy: metrics.val_accuracy || 0,
                valLoss: metrics.val_loss || 0
              });
            }
          }
          
          if (job.status === 'completed') {
            setTrainingProgress(100);
            setTrainingStage('completed');
            setTrainingStatus('Training completed!');
            setTrainingResult(job.result);
            toast({ title: "Training Completed", description: "AI model has been successfully trained for all students" });
            eventSource.close();
            eventSourceRef.current = null;
            setIsTraining(false);
          } else if (job.status === 'failed') {
            setTrainingStage('error');
            setTrainingStatus('Training failed');
            toast({ title: "Training Failed", description: job.error || "Failed to complete training", variant: "destructive" });
            eventSource.close();
            eventSourceRef.current = null;
            setIsTraining(false);
          }
        },
        (error) => {
          console.error('Training progress error:', error);
          setTrainingStage('error');
          setTrainingStatus('Connection error');
          toast({ title: "Connection Error", description: "Lost connection to training progress updates", variant: "destructive" });
          eventSource.close();
          eventSourceRef.current = null;
          setIsTraining(false);
        }
      );
      eventSourceRef.current = eventSource;
      
    } catch (error) {
      console.error('Training error:', error);
      setTrainingStage('error');
      setTrainingStatus('Training failed');
      setTrainingProgress(0);
      toast({
        title: "Error",
        description: "An unexpected error occurred during training",
        variant: "destructive",
      });
      setIsTraining(false);
    }
  };

  // Verification Functions
  const handleVerificationFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.files?.[0];
    const valid = raw ? validateFiles([raw]) : [];
    const file = valid[0];
    if (file) {
      if (verificationPreview) {
        URL.revokeObjectURL(verificationPreview);
      }
      setVerificationFile(file);
      try {
        let preview = '';
        const name = file.name.toLowerCase();
        if (file.type.startsWith('image/') && file.type !== 'image/tiff' && !name.endsWith('.tif') && !name.endsWith('.tiff')) {
          preview = URL.createObjectURL(file);
        } else {
          preview = await aiService.getPreviewURL(file);
        }
        setVerificationPreview(preview);
      } catch {
        setVerificationPreview(URL.createObjectURL(file));
      }
      setVerificationResult(null);
      markDirty();
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setUseCamera(true);
        markDirty();
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      toast({
        title: "Camera Error",
        description: "Unable to access camera. Please check permissions.",
        variant: "destructive",
      });
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const context = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      if (context) {
        context.drawImage(video, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'signature.png', { type: 'image/png' });
            setVerificationFile(file);
            setVerificationPreview(URL.createObjectURL(file));
            setUseCamera(false);
            setVerificationResult(null);
            markDirty();
            
            const stream = video.srcObject as MediaStream;
            stream?.getTracks().forEach(track => track.stop());
            setTimeout(() => {
              handleVerifySignature();
            }, 0);
          }
        });
      }
    }
  };

  const stopCamera = () => {
    if (videoRef.current) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream?.getTracks().forEach(track => track.stop());
      setUseCamera(false);
      markDirty();
    }
  };

  // Elapsed timer ticker
  useEffect(() => {
    let interval: number | undefined;
    if (isTraining) {
      interval = window.setInterval(() => {
        if (trainingStartTimeRef.current) {
          setElapsedMs(Date.now() - trainingStartTimeRef.current);
        }
      }, 1000);
    } else {
      setElapsedMs(0);
    }
    return () => {
      if (interval) window.clearInterval(interval);
    };
  }, [isTraining]);

  // Modal Functions
  type ModalContext = { kind: 'training', setType: 'genuine' | 'forged', cardId: string } | { kind: 'verification' } | null;
  const [modalContext, setModalContext] = useState<ModalContext>(null);
  
  const getModalFilename = (): string => {
    if (!isModalOpen) return '';
    if (modalContext?.kind === 'training') {
      const card = studentCards.find(c => c.id === modalContext.cardId);
      if (card) {
        const files = modalContext.setType === 'genuine' ? card.genuineFiles : card.forgedFiles;
        const idx = files.findIndex(f => f.preview === modalImages[modalImageIndex]);
        if (idx >= 0) {
          const f = files[idx];
          // Avoid showing long S3 URLs; show simple label
          return f.label ? `${f.label.toUpperCase()} sample` : 'Signature sample';
        }
        return '';
      }
    }
    if (modalContext?.kind === 'verification') {
      return verificationFile?.name || '';
    }
    return '';
  };

  const openImageModal = (images: string[], startIndex: number = 0, context: ModalContext = null) => {
    setModalImages(images);
    setModalImageIndex(startIndex);
    setIsModalOpen(true);
    setModalContext(context);
  };

  const closeImageModal = () => {
    setIsModalOpen(false);
    setModalImages([]);
    setModalImageIndex(0);
  };

  const goToPreviousImage = () => {
    setModalImageIndex(prev => prev > 0 ? prev - 1 : modalImages.length - 1);
  };

  const goToNextImage = () => {
    setModalImageIndex(prev => prev < modalImages.length - 1 ? prev + 1 : 0);
  };

  const deleteModalCurrentImage = () => {
    if (!modalContext || modalContext.kind !== 'training') return;
    const targetPreview = modalImages[modalImageIndex];
    const card = studentCards.find(c => c.id === modalContext.cardId);
    if (card) {
      if (modalContext.setType === 'genuine') {
        const idx = card.genuineFiles.findIndex(f => f.preview === targetPreview);
        if (idx !== -1) removeTrainingFile(idx, 'genuine', modalContext.cardId);
        const updated = card.genuineFiles.filter(f => f.preview !== targetPreview).map(f => f.preview);
        setModalImages(updated);
      } else {
        const idx = card.forgedFiles.findIndex(f => f.preview === targetPreview);
        if (idx !== -1) removeTrainingFile(idx, 'forged', modalContext.cardId);
        const updated = card.forgedFiles.filter(f => f.preview !== targetPreview).map(f => f.preview);
        setModalImages(updated);
      }
    }
    setModalImageIndex(prev => Math.max(0, prev - (modalImages.length === 1 ? 0 : 1)));
    if (modalImages.length <= 1) closeImageModal();
  };

  const handleVerifySignature = async () => {
    if (!verificationFile) {
      toast({
        title: "Error",
        description: "Please upload or capture a signature image",
        variant: "destructive",
      });
      return;
    }

    setIsVerifying(true);
    setVerificationResult(null);

    try {
      const result = await aiService.verifySignature(verificationFile);
      setVerificationResult(result);
      
      if (result.success) {
        toast({
          title: "Verification Complete",
          description: result.match 
            ? "Match found" 
            : "No match found",
        });
      } else {
        toast({
          title: "Verification Failed",
          description: result.message || "Failed to verify signature",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Verification error:', error);
      toast({
        title: "Error",
        description: "An unexpected error occurred during verification",
        variant: "destructive",
      });
    } finally {
      setIsVerifying(false);
    }
  };

  return (
    <Layout>
      <div
        className="flex-1 flex flex-col space-y-6 px-6 py-4"
        onClick={markDirty}
        onInput={markDirty}
      >
        {/* Page Header */}
        <div className="space-y-0.5">
          <h1 className="text-lg font-bold text-education-navy">SIGNATURE AI TRAINING & VERIFICATION</h1>
          <p className="text-sm text-muted-foreground">
            Train AI models for multiple students and verify signatures using machine learning
          </p>
        </div>

        {/* Model Training Card - Full Width */}
        <Card ref={studentContainerRef} className="w-full">
          <CardHeader>
            <div className="flex items-start justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  Model Training
                </CardTitle>
                <CardDescription>
                  Train AI models with uploaded signature data
                </CardDescription>
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <MoreVertical className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {!isViewingModels ? (
                    <>
                      <DropdownMenuItem disabled={isLocked} onClick={() => { if (!isLocked) addStudentCard(); }}>Add Student</DropdownMenuItem>
                      <DropdownMenuItem 
                        onClick={(e) => {
                          e.preventDefault();
                          handlePreventDuplicatesToggle(!preventDuplicates);
                        }}
                        className="flex items-center justify-between gap-2 cursor-pointer"
                      >
                        <span>Prevent duplicate uploads</span>
                        <Switch
                          checked={preventDuplicates}
                          onCheckedChange={handlePreventDuplicatesToggle}
                          onClick={(e) => e.stopPropagation()}
                        />
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem disabled={isLocked} onClick={async () => {
                        if (isLocked) return;
                        try {
                          const items = await aiService.listStudentsWithImages();
                          const byId = new Map(allStudents.map(s => [s.id, s]));
                          const toAdd = items
                            .map(it => byId.get(it.student_id))
                            .filter((s): s is Student => Boolean(s));
                          // Add cards for these students
                          let addedIds: number[] = [];
                          setStudentCards(prev => {
                            const existingIds = new Set(prev.filter(c => c.student).map(c => (c.student as any).id));
                            const newCards = toAdd
                              .filter(s => !existingIds.has(s.id))
                              .map(s => ({ id: `${Date.now()}-${s.id}`, student: s, genuineFiles: [], forgedFiles: [], isExpanded: true }));
                            addedIds = newCards.map(c => (c.student as any).id);
                            const merged = [...prev, ...newCards];
                            // Remove placeholder empty card if real students exist
                            const hasReal = merged.some(c => !!c.student);
                            return hasReal ? merged.filter(c => c.student) : merged;
                          });
                          // For each added student, load their images and populate previews
                          for (const sid of addedIds) {
                            try {
                              const persisted = await aiService.listSignatures(sid);
                              setStudentCards(prev => prev.map(c => {
                                if (c.student && c.student.id === sid) {
                                  const mapRec = (rec: any) => ({ file: new File([], rec.s3_url), preview: rec.s3_url, id: rec.id, s3Key: rec.s3_key, label: rec.label });
                                  return {
                                    ...c,
                                    genuineFiles: persisted.filter(x => x.label === 'genuine').map(mapRec),
                                    forgedFiles: persisted.filter(x => x.label === 'forged').map(mapRec),
                                  };
                                }
                                return c;
                              }));
                            } catch {}
                          }
                        } catch (e) {
                          toast({ title: 'Error', description: 'Failed to load students with images', variant: 'destructive' });
                        }
                      }}>Load students with images</DropdownMenuItem>
                      <DropdownMenuItem disabled={isLocked} onClick={async () => {
                        if (isLocked) return;
                        setIsViewingModels(true);
                        setIsLoadingModels(true);
                        setSelectedDate(null);
                        try {
                          const [individuals, globals] = await Promise.all([
                            aiService.getTrainedModels(),
                            aiService.getGlobalModels(),
                          ]);
                          const grouped: Record<string, { global?: TrainedModel; individual: TrainedModel[] }> = {};
                          for (const g of (globals as TrainedModel[])) {
                            const dateKey = (g.training_date || g.created_at || '').split('T')[0] || 'Unknown';
                            const bucket = grouped[dateKey] || { individual: [] };
                            bucket.global = g;
                            grouped[dateKey] = bucket;
                          }
                          for (const m of (individuals as TrainedModel[])) {
                            const dateKey = (m.training_date || m.created_at || '').split('T')[0] || 'Unknown';
                            const bucket = grouped[dateKey] || { individual: [] };
                            bucket.individual.push(m);
                            grouped[dateKey] = bucket;
                          }
                          setDateGroupedModels(grouped);
                        } catch (error) {
                          setDateGroupedModels({});
                        }
                        setIsLoadingModels(false);
                      }}>View Models</DropdownMenuItem>
                    </>
                  ) : (
                    <>
                      <DropdownMenuItem onClick={() => setIsViewingModels(false)}>Train</DropdownMenuItem>
                    </>
                  )}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Student Training Cards / Trained Models - Shared container */}
            <div className={`grid ${isViewingModels ? 'grid-cols-2' : 'grid-cols-2 md:grid-cols-3 lg:grid-cols-6'} gap-2 min-h-[424px] max-h-[424px] overflow-auto pr-1 content-start justify-start auto-rows-[40px] items-start`}>
              {!isViewingModels ? (
                <>
                  {studentCards.map((card, index) => (
                    <div key={card.id}>
                      <StudentTrainingCard
                        card={card}
                        index={index}
                        onUpdate={(updates) => updateStudentCard(card.id, updates)}
                        onRemove={() => removeStudentCard(card.id)}
                        onOpenStudentDialog={() => openStudentDialog(card.id)}
                        onTrainingFilesChange={handleTrainingFilesChange}
                        onRemoveTrainingFile={removeTrainingFile}
                        onOpenImageModal={openImageModal}
                        onRemoveAllSamples={removeAllSamples}
                        onCardClick={() => openStudentFormDialog(card.id)}
                        locked={isLocked}
                      />
                    </div>
                  ))}
                  {/* Add Student Placeholder */}
                  <div className={`border-2 border-dashed border-gray-300 rounded-lg ${isLocked ? 'opacity-50 cursor-not-allowed' : 'hover:border-gray-400 cursor-pointer'} transition-colors h-10 flex items-center justify-center px-2`} onClick={() => { if (!isLocked) addStudentCard(); }}>
                    <div className="text-sm text-gray-500 hover:text-gray-600">
                      Add Student
                    </div>
                  </div>
                </>
              ) : (
                <>
                  {isLoadingModels ? (
                    <div className="col-span-2 flex items-center justify-center text-sm text-muted-foreground">Loading models...</div>
                  ) : selectedDate ? (
                    // Detailed view for selected date
                    <div className="col-span-2 space-y-4">
                      <div className="flex items-center gap-2 mb-4">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelectedDate(null)}
                          className="flex items-center gap-1"
                        >
                          <ChevronLeft className="w-4 h-4" />
                          Back to dates
                        </Button>
                        <h3 className="text-sm font-semibold">Models trained on {selectedDate}</h3>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2">
                        {/* Global Model */}
                        {dateGroupedModels[selectedDate]?.global && (
                          <div className="group">
                            <Card className="h-10 border-blue-200 bg-blue-50">
                              <CardContent className="py-0 px-2 pr-1 h-full flex items-center">
                                <div className="flex-1 min-w-0 text-xs">
                                  <div className="font-medium truncate text-blue-800">🌐 Global Model</div>
                                  <div className="text-[10px] text-blue-600 truncate">{dateGroupedModels[selectedDate].global?.model_path || 'n/a'}</div>
                                </div>
                                <div className="text-[10px] text-blue-600 mr-2 whitespace-nowrap">{selectedDate}</div>
                                <div className="text-[10px] font-medium mr-2 whitespace-nowrap text-blue-800">
                                  {dateGroupedModels[selectedDate].global?.accuracy ? `${Math.round(dateGroupedModels[selectedDate].global!.accuracy * 100)}%` : ''}
                                </div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-5 w-5 p-0 opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground hover:bg-transparent"
                                  onClick={() => setConfirmDeleteModelId(dateGroupedModels[selectedDate].global?.id)}
                                  aria-label="Delete global model"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </CardContent>
                            </Card>
                          </div>
                        )}
                        
                        {/* Individual Models */}
                        {dateGroupedModels[selectedDate]?.individual.map((m: TrainedModel) => (
                          <div key={m.id} className="group">
                            <Card className="h-10">
                              <CardContent className="py-0 px-2 pr-1 h-full flex items-center">
                                <div className="flex-1 min-w-0 text-xs">
                                  <div className="font-medium truncate">{m.student_name || m.student?.full_name || m.student_full_name || 'Unknown Student'}</div>
                                  <div className="text-[10px] text-muted-foreground truncate">{m.model_path || m.model?.path || m.artifact_path || 'n/a'}</div>
                                </div>
                                <div className="text-[10px] text-muted-foreground mr-2 whitespace-nowrap">{selectedDate}</div>
                                <div className="text-[10px] font-medium mr-2 whitespace-nowrap">{m.accuracy ? `${Math.round(m.accuracy * 100)}%` : ''}</div>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-5 w-5 p-0 opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground hover:bg-transparent"
                                  onClick={() => setConfirmDeleteModelId(m.id)}
                                  aria-label="Delete model"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </Button>
                              </CardContent>
                            </Card>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : Object.keys(dateGroupedModels).length === 0 ? (
                    <div className="col-span-2 flex items-center justify-center text-sm text-muted-foreground">No trained models yet.</div>
                  ) : (
                    // Date overview cards
                    Object.entries(dateGroupedModels).map(([date, models]) => (
                      <div key={date} className="group">
                        <Card 
                          className="h-10 cursor-pointer hover:shadow-md transition-shadow"
                          onClick={() => setSelectedDate(date)}
                        >
                          <CardContent className="py-0 px-2 pr-1 h-full flex items-center">
                            <div className="flex-1 min-w-0 text-xs">
                              <div className="font-medium truncate">{date}</div>
                              <div className="text-[10px] text-muted-foreground truncate">
                                {models.global ? 'Global + ' : ''}{models.individual.length} individual model{models.individual.length !== 1 ? 's' : ''}
                              </div>
                            </div>
                            <div className="text-[10px] text-muted-foreground mr-2 whitespace-nowrap">
                              {models.global ? '🌐' : ''} {models.individual.length} 📊
                            </div>
                            <ChevronRight className="w-3 h-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                          </CardContent>
                        </Card>
                      </div>
                    ))
                  )}
                </>
              )}
            </div>

            {/* Delete Model Confirmation Dialog */}
            <Dialog open={confirmDeleteModelId !== null} onOpenChange={(open) => { if (!open) setConfirmDeleteModelId(null); }}>
              <DialogContent className="max-w-sm">
                <DialogHeader>
                  <DialogTitle>Delete Trained Model?</DialogTitle>
                </DialogHeader>
                <div className="text-sm text-muted-foreground">
                  Are you sure you want to delete this trained model? You will need to retrain to recreate it.
                </div>
                <div className="flex justify-end gap-2 pt-2">
                  <Button variant="outline" size="sm" onClick={() => setConfirmDeleteModelId(null)}>Cancel</Button>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => {
                      if (confirmDeleteModelId === null) return;
                      
                      // Handle deletion for both legacy and new hybrid view
                      if (selectedDate && dateGroupedModels[selectedDate]) {
                        // Delete from hybrid view
                        setDateGroupedModels(prev => {
                          const updated = { ...prev };
                          if (updated[selectedDate]) {
                            if (updated[selectedDate].global?.id === confirmDeleteModelId) {
                              updated[selectedDate] = { ...updated[selectedDate], global: undefined };
                            } else {
                              updated[selectedDate] = {
                                ...updated[selectedDate],
                                individual: updated[selectedDate].individual.filter(m => m.id !== confirmDeleteModelId)
                              };
                            }
                          }
                          return updated;
                        });
                      } else {
                        // Delete from legacy view
                        setTrainedModels(prev => prev.filter(x => x.id !== confirmDeleteModelId));
                      }
                      
                      setConfirmDeleteModelId(null);
                    }}
                  >
                    Delete
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            {/* Training Progress (moved below button) */}

            {/* Training Results */}
            {trainingResult && (
              <div>
                <Alert className={trainingResult.success ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
                  <div className="flex items-center gap-2">
                    {trainingResult.success ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-red-600" />
                    )}
                    <AlertDescription className="text-sm">
                      <strong>{trainingResult.success ? 'Training Completed' : 'Training Failed'}</strong>
                      <p className="text-xs text-muted-foreground mt-1">
                        {trainingResult.message}
                      </p>
                    </AlertDescription>
                  </div>
                </Alert>
              </div>
            )}

            {/* Train Model Button at Bottom */}
            <div className="pt-4 border-t">
              <div className="flex flex-col items-center space-y-4">
                {/* Hybrid mode only + GPU toggle */}
                {!isViewingModels && (
                  <div className="flex flex-col items-center gap-3 text-sm">
                    <div className="flex items-center gap-2">
                      {/* Intentionally no visible training mode label */}
                    </div>
                    {/* GPU Training Toggle */}
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="use-gpu"
                        checked={useGPU}
                        onChange={(e) => { if (!isLocked) setUseGPU(e.target.checked); }}
                        disabled={isLocked}
                        className="rounded border-gray-300"
                      />
                      <Label htmlFor="use-gpu" className="text-muted-foreground">
                        🚀 Use AWS GPU (10-50x faster)
                      </Label>
                    </div>
                  </div>
                )}
                
                <Button
                  onClick={handleTrainModel}
                  disabled={isViewingModels || !canTrainModel() || isTraining}
                  className="w-full max-w-[280px]"
                  size="lg"
                >
                  {isTraining ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Training... {Math.round(trainingProgress)}%
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 mr-2" />
                      Train Model
                    </>
                  )}
                </Button>
                
                <div className="text-center text-sm text-muted-foreground">
                  {studentCards.filter(c => c.student).length} students • {getTotalTrainingData().genuine + getTotalTrainingData().forged} samples ready
                  {/* Hybrid behavior is implicit; no extra label */}
                </div>
                <div className="text-center text-xs text-muted-foreground">
                  {preventDuplicates ? '🛡️ Duplicate prevention enabled' : '⚠️ Duplicate prevention disabled'}
                </div>
              </div>
            </div>

            {isTraining && (
              <div className="space-y-3 mt-4">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Progress: {Math.round(trainingProgress)}%</span>
                  <span>{estimatedTimeRemaining || ''}</span>
                </div>
                <Progress value={trainingProgress} className="w-full" />
                {trainingLogs.length > 0 && (
                  <div className="bg-gray-900 text-green-400 rounded-lg p-3 font-mono text-xs max-h-32 overflow-y-auto">
                    <div className="text-green-300 text-sm font-semibold mb-2">Training Progress</div>
                    {trainingLogs.slice(-5).map((log, index) => (
                      <div key={index} className="mb-1">
                        {log}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Verification Card - Below Model Training */}
        <div className="mt-6">
          <Card ref={verificationCardRef} className="h-fit">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="flex items-center gap-2">
                      <Scan className="w-5 h-5" />
                      Signature Verification
                    </CardTitle>
                    <CardDescription>
                      Upload or capture a signature to verify against trained models
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-4">
                    {/* Camera/Upload Toggle */}
                    <div className="flex gap-2">
                  <Button
                    variant={useCamera ? "default" : "outline"}
                    size="sm"
                    onClick={startCamera}
                    className="flex items-center gap-2 hover:bg-transparent hover:text-foreground"
                  >
                    <Camera className="w-4 h-4" />
                    Camera
                  </Button>
                  <Button
                    variant={!useCamera ? "default" : "outline"}
                    size="sm"
                    onClick={() => {
                      setUseCamera(false);
                      stopCamera();
                      verificationInputRef.current?.click();
                    }}
                    className="flex items-center gap-2"
                  >
                    <Upload className="w-4 h-4" />
                    Upload
                  </Button>
                    </div>

                    {/* Large Square Preview Box */}
                    <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Signature Preview</Label>
                    <div className="text-xs text-muted-foreground">{useCamera ? 'Camera' : 'Upload'}</div>
                  </div>
                  <div className="w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50">
                    {useCamera ? (
                      <div className="w-full h-full">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          className="w-full h-full object-cover rounded-lg"
                        />
                      </div>
                    ) : verificationPreview ? (
                      <img
                        src={verificationPreview}
                        alt="Signature preview"
                        className="w-full h-full object-contain rounded-lg cursor-pointer hover:opacity-80 transition-opacity"
                        onClick={() => openImageModal([verificationPreview], 0)}
                      />
                    ) : (
                      <div className="text-center text-gray-500">
                        <Upload className="w-8 h-8 mx-auto mb-2" />
                        <p>No signature selected</p>
                      </div>
                    )}
                      </div>
                    </div>

                    {/* Camera Controls */}
                    {useCamera && (
                      <div className="flex gap-2">
                        <Button onClick={capturePhoto} size="sm" className="flex-1">
                          Capture
                        </Button>
                        <Button onClick={stopCamera} variant="outline" size="sm" className="flex-1">
                          Cancel
                        </Button>
                      </div>
                    )}

                    {/* Hidden File Input */}
                    <Input
                      ref={verificationInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleVerificationFileChange}
                      className="hidden"
                    />

                    {/* Verify Button */}
                    <Button
                      onClick={handleVerifySignature}
                      disabled={!verificationFile || isVerifying}
                      className="w-full"
                    >
                      {isVerifying ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Verifying...
                        </>
                      ) : (
                        <>
                          <Scan className="w-4 h-4 mr-2" />
                          Verify Signature
                        </>
                      )}
                    </Button>
                  </div>

                  <div className="space-y-3 md:pl-2 md:border-l md:border-border">
                    <div className="text-sm font-semibold">Verification Result</div>
                    {verificationResult ? (
                      <Alert className={
                        verificationResult.match 
                          ? "border-green-200 bg-green-50" 
                          : "border-red-200 bg-red-50"
                      }>
                        <div className="flex items-start gap-2">
                          {verificationResult.match ? (
                            <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                          ) : (
                            <XCircle className="w-4 h-4 text-red-600 mt-0.5" />
                          )}
                          <AlertDescription>
                            <div className="space-y-2">
                              <p>
                                <strong>Result:</strong> {
                                  verificationResult.match 
                                    ? 'Match Found' 
                                    : 'No Match'
                                }
                              </p>
                              <p>
                                <strong>Confidence:</strong> {
                                  `${(verificationResult.score * 100).toFixed(1)}%`
                                }
                              </p>
                              {verificationResult.predicted_student && (
                                <p>
                                  <strong>Predicted Student:</strong> {verificationResult.predicted_student.firstname} {verificationResult.predicted_student.surname}
                                </p>
                              )}
                              {verificationResult.message && (
                                <p className="text-sm text-muted-foreground">
                                  {verificationResult.message}
                                </p>
                              )}
                            </div>
                          </AlertDescription>
                        </div>
                      </Alert>
                    ) : (
                      <div className="text-xs text-muted-foreground">Run a verification to see results here.</div>
                    )}
                  </div>
                </div>
              </CardContent>
          </Card>
        </div>

        {/* Image Preview Modal */}
        <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
          <DialogContent className="max-w-4xl max-h-[90vh] p-0">
            <DialogHeader className="p-6 pb-0">
              <DialogTitle>
                <div className="flex items-center justify-between">
                  <span>Image Preview</span>
                  {getModalFilename() && (
                    <span className="text-xs text-muted-foreground truncate max-w-[60%]" title={getModalFilename()}>
                      {getModalFilename()}
                    </span>
                  )}
                </div>
              </DialogTitle>
            </DialogHeader>
            <div className="relative p-6">
              {modalImages.length > 0 && (
                <>
                  <div className="relative group">
                    <img
                      src={modalImages[modalImageIndex]}
                      alt={`Preview ${modalImageIndex + 1}`}
                      className="w-full h-auto max-h-[60vh] object-contain mx-auto"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="hidden group-hover:flex absolute left-3 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full w-8 h-8 items-center justify-center"
                      onClick={goToPreviousImage}
                      aria-label="Previous"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="hidden group-hover:flex absolute right-3 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/70 text-white rounded-full w-8 h-8 items-center justify-center"
                      onClick={goToNextImage}
                      aria-label="Next"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </Button>
                  </div>

                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
                    <div className="bg-black/50 text-white px-3 py-1 rounded-full text-sm flex items-center gap-3">
                      <span>{modalImageIndex + 1} / {modalImages.length}</span>
                      {modalContext && modalContext.kind === 'training' && (
                        <button onClick={deleteModalCurrentImage} aria-label="Delete Image" className="text-white">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          </DialogContent>
        </Dialog>

        {/* Student Selection Dialog */}
        <Dialog open={isStudentDialogOpen} onOpenChange={setIsStudentDialogOpen}>
          <DialogContent className="max-w-6xl w-full h-[85vh]">
            <DialogHeader>
              <DialogTitle>Select Student</DialogTitle>
            </DialogHeader>
            <div className="flex flex-col h-full gap-4">
              <div className="flex items-center justify-between gap-3">
                <Input
                  placeholder="Search by ID or Name"
                  value={studentSearch}
                  onChange={(e) => setStudentSearch(e.target.value)}
                  className="w-full max-w-sm"
                />
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
                    <input
                      type="checkbox"
                      className="h-4 w-4"
                      checked={showSelectedStudents}
                      onChange={(e) => {
                        setShowSelectedStudents(e.target.checked);
                        setStudentPage(1);
                      }}
                    />
                    Show selected
                  </label>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (allSelectedOnPage) {
                        setSelectedStudentIds(prev => {
                          const next = new Set(prev);
                          selectablePagedStudentIds.forEach(id => next.delete(id));
                          return next;
                        });
                      } else {
                        setSelectedStudentIds(prev => {
                          const next = new Set(prev);
                          selectablePagedStudentIds.forEach(id => next.add(id));
                          return next;
                        });
                      }
                    }}
                  >
                    {allSelectedOnPage ? 'Unselect All (page)' : 'Select All (page)'}
                  </Button>
                </div>
              </div>
              <div className="overflow-auto border rounded-md h-[60vh]">
                {isLoadingStudents ? (
                  <div className="p-4 text-sm text-muted-foreground">Loading students…</div>
                ) : isStudentSearching ? (
                  <div className="p-4 text-sm text-muted-foreground">Searching…</div>
                ) : visibleStudents.length === 0 ? (
                  <div className="p-4 text-sm text-muted-foreground">No results</div>
                ) : (
                  <ul className="divide-y">
                    {pagedStudents.map((s) => {
                      const isAssignedElsewhere = selectedElsewhereIds.has(s.id);
                      const checked = selectedStudentIds.has(s.id);
                      return (
                        <li key={s.id}>
                          <label className={`flex items-center justify-between p-2 gap-2 ${isAssignedElsewhere ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}>
                            <div className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                className="h-4 w-4"
                                disabled={isAssignedElsewhere}
                                checked={checked}
                                onChange={(e) => {
                                  const isChecked = e.target.checked;
                                  setSelectedStudentIds(prev => {
                                    const next = new Set(prev);
                                    if (isChecked) next.add(s.id); else next.delete(s.id);
                                    return next;
                                  });
                                }}
                              />
                              <div>
                                <div className="font-medium text-sm">{`${s.firstname} ${s.surname}`}</div>
                                <div className="text-[11px] text-muted-foreground">ID: {s.student_id} • {s.program} • Year {s.year} • Sec {s.section}</div>
                              </div>
                            </div>
                            {isAssignedElsewhere && (
                              <Badge variant="secondary" className="text-xs">In use</Badge>
                            )}
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" disabled={studentPage <= 1} onClick={() => setStudentPage(p => Math.max(1, p - 1))}>Previous</Button>
                  <Button variant="outline" size="sm" disabled={studentPage >= totalStudentPages} onClick={() => setStudentPage(p => Math.min(totalStudentPages, p + 1))}>Next</Button>
                </div>
                <div className="flex-1 text-center text-xs text-muted-foreground">Page {studentPage} of {totalStudentPages}</div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setIsStudentDialogOpen(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={() => {
                      if (studentSelectMode === 'bulkAdd') {
                        if (selectedStudentIds.size === 0) return setIsStudentDialogOpen(false);
                        const selected = allStudents.filter(s => selectedStudentIds.has(s.id) && !selectedElsewhereIds.has(s.id));
                        setStudentCards(prev => {
                          let base = prev;
                          if (
                            prev.length === 1 &&
                            prev[0].student === null &&
                            prev[0].genuineFiles.length === 0 &&
                            prev[0].forgedFiles.length === 0
                          ) {
                            base = [];
                          }
                          return ([
                            ...base,
                            ...selected.map(s => ({ id: `${Date.now()}-${s.id}`, student: s, genuineFiles: [], forgedFiles: [], isExpanded: true }))
                          ]);
                        });
                        setIsStudentDialogOpen(false);
                      } else {
                        const chosen = allStudents.find(s => selectedStudentIds.has(s.id) && !selectedElsewhereIds.has(s.id));
                        if (chosen && currentCardId) {
                          handleStudentSelection(chosen);
                          setIsStudentDialogOpen(false);
                        }
                      }
                    }}
                    disabled={selectedStudentIds.size === 0}
                  >
                    {studentSelectMode === 'bulkAdd' ? 'Add Selected' : 'Assign'}
                  </Button>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Student Form Dialog - Original Expanded Card Structure */}
        <Dialog open={isStudentFormDialogOpen} onOpenChange={setIsStudentFormDialogOpen}>
          <DialogContent className="max-w-2xl h-[90vh] overflow-y-auto p-6 pt-4">
            <DialogHeader className="p-0 m-0 !space-y-0">
              <DialogTitle className="mb-0">Student Training Details</DialogTitle>
            </DialogHeader>
            <div className="flex flex-col gap-0 h-full">
              {currentFormCardId && (() => {
                const card = studentCards.find(c => c.id === currentFormCardId);
                if (!card) return null;
                
                return (
                  <>
                    {/* Student Info - Original Grid Layout */}
                    {card.student ? (
                      <div className="space-y-0">
                        {/* Student Name */}
                        <div>
                          <div className="text-base font-semibold">
                            Name: {card.student.firstname} {card.student.surname}
                          </div>
                        </div>
                        {/* Student Details Grid */}
                        <div className="grid grid-cols-2 gap-1 text-xs">
                          <div>
                            <div className="font-medium">ID: {card.student.student_id}</div>
                          </div>
                          <div>
                            <div className="font-medium">Program: {card.student.program}</div>
                          </div>
                          <div>
                            <div className="font-medium">Year: {card.student.year}</div>
                          </div>
                          <div>
                            <div className="font-medium">Section: {card.student.section}</div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <User className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                        <p className="text-sm text-muted-foreground mb-4">No student selected</p>
                        <Button
                          onClick={() => {
                            openStudentDialog(card.id);
                          }}
                          size="lg"
                          variant="outline"
                          className="flex items-center gap-2 mx-auto"
                        >
                          <User className="w-4 h-4" />
                          Select Student
                        </Button>
                      </div>
                    )}

                    {/* Upload Buttons - Original Layout */}
                    {card.student && (
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex items-center gap-2 hover:bg-transparent hover:text-foreground"
                          onClick={() => {
                            const input = document.createElement('input');
                            input.type = 'file';
                            input.accept = 'image/*';
                            input.multiple = true;
                            input.onchange = (e) => {
                              const files = Array.from((e.target as HTMLInputElement).files || []);
                              handleTrainingFilesChange(files, 'forged', card.id);
                            };
                            input.click();
                          }}
                        >
                          <Upload className="w-4 h-4" />
                          Forged
                        </Button>
                        <Button
                          variant="default"
                          size="sm"
                          className="flex items-center gap-2"
                          onClick={() => {
                            const input = document.createElement('input');
                            input.type = 'file';
                            input.accept = 'image/*';
                            input.multiple = true;
                            input.onchange = (e) => {
                              const files = Array.from((e.target as HTMLInputElement).files || []);
                              handleTrainingFilesChange(files, 'genuine', card.id);
                            };
                            input.click();
                          }}
                        >
                          <Upload className="w-4 h-4" />
                          Genuine
                        </Button>
                      </div>
                    )}

                    {/* Training Images Preview - Expands to fill space with compact grid and toggles */}
                    {card.student && (
                      <div className="flex flex-col gap-2 flex-1 min-h-0">
                        <div className="flex items-center justify-between">
                          <Label className="text-sm">Training Images</Label>
                          <div className="text-xs text-muted-foreground">
                            <span className={`${trainingImagesSet==='genuine' ? 'font-semibold' : ''}`}>Genuine ({card.genuineFiles.length})</span>
                            <span className="mx-1">/</span>
                            <span className={`${trainingImagesSet==='forged' ? 'font-semibold' : ''}`}>Forged ({card.forgedFiles.length})</span>
                          </div>
                        </div>
                        <div className="relative w-full h-[480px] border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 group overflow-hidden">
                          {((trainingImagesSet === 'genuine') ? card.genuineFiles.length : card.forgedFiles.length) > 0 ? (
                            <div className="grid grid-cols-4 gap-2 w-full h-full p-2 overflow-y-auto">
                              {((trainingImagesSet === 'genuine') ? card.genuineFiles : card.forgedFiles).map((item, index) => (
                                <div 
                                  key={index} 
                                  className="relative group/itm cursor-pointer pb-[100%] overflow-hidden rounded-md border border-border"
                                  onClick={() => openImageModal(
                                    ((trainingImagesSet === 'genuine') ? card.genuineFiles : card.forgedFiles).map(f => f.preview), 
                                    index, 
                                    { kind: 'training', setType: trainingImagesSet, cardId: card.id }
                                  )}
                                >
                                  <img
                                    src={item.preview}
                                    alt={item.label ? `${item.label} sample ${index + 1}` : `Sample ${index + 1}`}
                                    className="absolute inset-0 w-full h-full object-cover hover:opacity-80 transition-opacity"
                                    loading="lazy"
                                  />
                                  <button
                                    type="button"
                                    className="absolute top-1 right-1 bg-black/50 hover:bg-black/70 text-white rounded p-1 opacity-0 group-hover/itm:opacity-100 transition-opacity"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      removeTrainingFile(index, trainingImagesSet, card.id);
                                    }}
                                    aria-label="Delete image"
                                  >
                                    <Trash2 className="w-4 h-4" />
                                  </button>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="h-full flex flex-col items-center justify-center text-gray-500">
                              <Upload className="w-6 h-6 mb-1" />
                              <p className="text-xs">No images</p>
                            </div>
                          )}
                          {/* Chevron controls on hover to toggle between sets */}
                          <button
                            className="hidden group-hover:flex absolute left-2 top-1/2 -translate-y-1/2 bg-black/40 hover:bg-black/60 text-white rounded-full w-8 h-8 items-center justify-center"
                            onClick={() => setTrainingImagesSet(prev => prev === 'genuine' ? 'forged' : 'genuine')}
                            aria-label="Previous"
                            type="button"
                          >
                            <ChevronLeft className="w-4 h-4" />
                          </button>
                          <button
                            className="hidden group-hover:flex absolute right-2 top-1/2 -translate-y-1/2 bg-black/40 hover:bg-black/60 text-white rounded-full w-8 h-8 items-center justify-center"
                            onClick={() => setTrainingImagesSet(prev => prev === 'genuine' ? 'forged' : 'genuine')}
                            aria-label="Next"
                            type="button"
                          >
                            <ChevronRight className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Data Summary removed as requested */}
                  </>
                );
              })()}
            </div>
          </DialogContent>
        </Dialog>

        {/* Unsaved Changes Dialog */}
        <UnsavedChangesDialog
          open={showConfirmDialog}
          onConfirm={() => {
            confirmClose();
            setPendingNavigation(null);
          }}
          onCancel={() => {
            cancelClose();
            setPendingNavigation(null);
          }}
        />
      </div>
    </Layout>
  );
};

export default SignatureAI;