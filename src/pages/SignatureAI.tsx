import React, { useEffect, useRef, useState, useLayoutEffect } from 'react';
import Layout from '@/components/Layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { FileUpload } from '@/components/ui/file-upload';
import { Separator } from '@/components/ui/separator';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
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
  MoreVertical,
  Trash2,
  AlertTriangle
} from 'lucide-react';
import { aiService, AI_CONFIG } from '@/lib/aiService';
import { fetchStudents } from '@/lib/supabaseService';
import type { Student } from '@/types';
import { Progress } from '@/components/ui/progress';

interface TrainingFile {
  file: File;
  preview: string;
}

type TrainingCard = {
  id: string;
  selectedStudent: Student | null;
  genuineFiles: TrainingFile[];
  forgedFiles: TrainingFile[];
};

const SignatureAI = () => {
  const { toast } = useToast();
  const pageRef = useRef<HTMLDivElement>(null);
  
  // Training Section State (multi-card)
  const [cards, setCards] = useState<TrainingCard[]>([{
    id: crypto.randomUUID(),
    selectedStudent: null,
    genuineFiles: [],
    forgedFiles: [],
  }]);
  // Focused card context for uploads and UI toggles
  const [activeCardId, setActiveCardId] = useState<string>(() => cards[0]?.id || '');
  const getActiveCard = () => cards.find(c => c.id === activeCardId) || cards[0];
  const [currentTrainingSet, setCurrentTrainingSet] = useState<'genuine' | 'forged'>('genuine');
  const [studentDialogForCardId, setStudentDialogForCardId] = useState<string | null>(null);
  const centerButtonRef = useRef<HTMLButtonElement | null>(null);
  const verificationCardRef = useRef<HTMLDivElement | null>(null);
  const leftCardRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [connectorPaths, setConnectorPaths] = useState<{ id: string; d: string }[]>([]);
  const [verifyConnector, setVerifyConnector] = useState<string>('');
  const [cardProgress, setCardProgress] = useState<Record<string, number>>({});
  const [globalProgress, setGlobalProgress] = useState<number>(0);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<any>(null);
  
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
  const [verificationResult, setVerificationResult] = useState<any>(null);
  const [useCamera, setUseCamera] = useState(false);
  const [referenceFiles, setReferenceFiles] = useState<File[]>([]);
  
  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalImageIndex, setModalImageIndex] = useState(0);
  const [modalImages, setModalImages] = useState<string[]>([]);
  
  // Dropdown State
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isConfirmRemoveOpen, setIsConfirmRemoveOpen] = useState(false);
  const [visibleCounts, setVisibleCounts] = useState<{ genuine: number; forged: number }>({ genuine: 60, forged: 60 });
  const STORAGE_KEY = 'signatureAIState:v2-multicard';
  const NAVIGATION_FLAG = 'signatureAIInternalNav:v1';

  // Unsaved changes handling
  const navigate = useNavigate();
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
      // Execute the pending navigation when user confirms
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
          handleClose(); // This will show the unsaved changes dialog
        }
      }
    };

    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, [hasUnsavedChanges, location.pathname, handleClose]);

  type SerializableStudent = Pick<Student, 'id' | 'student_id' | 'firstname' | 'surname' | 'program' | 'year' | 'section'>;
  type SerializableTraining = { name: string; type: string; size: number };

  // No longer serializing file contents to avoid storage quota issues

  // No deserialization from data URLs

  const saveSessionState = async (extra?: { addGenuine?: File[]; addForged?: File[] }) => {
    try {
      // Prepare serializable lists. Append newly added files efficiently.
      const genuineSerial: SerializableTraining[] = [];
      for (const item of (getActiveCard()?.genuineFiles || [])) {
        genuineSerial.push({
          name: item.file.name,
          type: item.file.type,
          size: item.file.size,
        });
      }
      const forgedSerial: SerializableTraining[] = [];
      for (const item of (getActiveCard()?.forgedFiles || [])) {
        forgedSerial.push({
          name: item.file.name,
          type: item.file.type,
          size: item.file.size,
        });
      }
      const studentSerial: SerializableStudent | null = (getActiveCard()?.selectedStudent as SerializableStudent) 
        ? {
            id: (getActiveCard()!.selectedStudent as Student).id,
            student_id: (getActiveCard()!.selectedStudent as Student).student_id,
            firstname: (getActiveCard()!.selectedStudent as Student).firstname,
            surname: (getActiveCard()!.selectedStudent as Student).surname,
            program: (getActiveCard()!.selectedStudent as Student).program,
            year: (getActiveCard()!.selectedStudent as Student).year,
            section: (getActiveCard()!.selectedStudent as Student).section,
          }
        : null;
      const payload = {
        cards: cards.map(c => ({
          id: c.id,
          selectedStudent: c.selectedStudent ? {
            id: c.selectedStudent.id,
            student_id: c.selectedStudent.student_id,
            firstname: c.selectedStudent.firstname,
            surname: c.selectedStudent.surname,
            program: c.selectedStudent.program,
            year: c.selectedStudent.year,
            section: c.selectedStudent.section,
          } : null,
          genuine: c.genuineFiles.map(f => ({ name: f.file.name, type: f.file.type, size: f.file.size })),
          forged: c.forgedFiles.map(f => ({ name: f.file.name, type: f.file.type, size: f.file.size })),
        })),
        selectedStudent: studentSerial,
        currentTrainingSet,
        visibleCounts,
        genuine: genuineSerial,
        forged: forgedSerial,
        timestamp: Date.now(), // Add timestamp for debugging
      };
      const json = JSON.stringify(payload);
      if (json.length < 1500000) {
        sessionStorage.setItem(STORAGE_KEY, json);
      }
      // Mark that we're navigating internally (not refreshing)
      sessionStorage.setItem(NAVIGATION_FLAG, 'true');
      console.log('State saved successfully', { timestamp: payload.timestamp });
    } catch (e) {
      console.warn('Failed saving session state', e);
    }
  };

  const loadSessionState = async () => {
    try {
      // Check if we have a navigation flag indicating internal navigation
      const hasNavigationFlag = sessionStorage.getItem(NAVIGATION_FLAG);
      
      // Only clear state on actual browser refresh/restart, preserve for internal navigation
      if (!hasNavigationFlag && window.performance.navigation && window.performance.navigation.type === 1) {
        // This is a browser refresh without navigation flag - clear state for fresh start
        console.log('Page reload detected, clearing saved state');
        sessionStorage.removeItem(STORAGE_KEY);
        sessionStorage.removeItem(NAVIGATION_FLAG);
        return;
      }
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) {
        console.log('No saved state found');
        return;
      }
      console.log('Loading saved state...');
      const parsed = JSON.parse(raw) as {
        selectedStudent: SerializableStudent | null;
        currentTrainingSet: 'genuine' | 'forged';
        visibleCounts: { genuine: number; forged: number };
        genuine: SerializableTraining[];
        forged: SerializableTraining[];
        timestamp?: number;
      };
      console.log('State loaded successfully', { 
        timestamp: parsed.timestamp,
        studentSelected: !!parsed.selectedStudent,
        genuineCount: parsed.genuine?.length || 0,
        forgedCount: parsed.forged?.length || 0 
      });
      if (parsed.selectedStudent) {
        setSelectedStudent(parsed.selectedStudent as unknown as Student);
      }
      setVisibleCounts(parsed.visibleCounts || { genuine: 60, forged: 60 });
      setCurrentTrainingSet(parsed.currentTrainingSet || 'genuine');
      // Do not rehydrate file blobs from storage
      
      // Clear navigation flag after successful load
      sessionStorage.removeItem(NAVIGATION_FLAG);
    } catch (e) {
      console.warn('Failed loading session state', e);
    }
  };

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

  // Student Selection State
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const [isStudentDialogOpen, setIsStudentDialogOpen] = useState(false);
  const [studentSearch, setStudentSearch] = useState('');
  const [isStudentCollapsed, setIsStudentCollapsed] = useState(true);
  const [debouncedStudentSearch, setDebouncedStudentSearch] = useState('');
  const [isStudentSearching, setIsStudentSearching] = useState(false);
  const [allStudents, setAllStudents] = useState<Student[]>([]);
  const [isLoadingStudents, setIsLoadingStudents] = useState(false);
  
  // Student switch confirmation state
  const [isStudentSwitchConfirmOpen, setIsStudentSwitchConfirmOpen] = useState(false);
  const [pendingStudent, setPendingStudent] = useState<Student | null>(null);

  // Fetch students on component mount
  React.useEffect(() => {
    const loadStudents = async () => {
      setIsLoadingStudents(true);
      try {
        const students = await fetchStudents();
        // Sort alphabetically by name (firstname + surname)
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
  }, []);

  // Restore persisted state on mount
  React.useEffect(() => {
    loadSessionState();
  }, []);

  // Persist on critical state changes
  React.useEffect(() => {
    saveSessionState();
  }, [cards, currentTrainingSet, visibleCounts]);

  // Ensure state is persisted on internal navigation or tab hide
  React.useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        saveSessionState();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      saveSessionState();
    };
  }, [cards, currentTrainingSet, visibleCounts]);

  React.useEffect(() => {
    setIsStudentSearching(true);
    const t = setTimeout(() => {
      setDebouncedStudentSearch(studentSearch.trim());
      setIsStudentSearching(false);
    }, 300);
    return () => clearTimeout(t);
  }, [studentSearch]);

  const filteredStudents = debouncedStudentSearch 
    ? allStudents.filter((s) => (
        s.student_id.includes(debouncedStudentSearch) ||
        `${s.firstname} ${s.surname}`.toLowerCase().includes(debouncedStudentSearch.toLowerCase())
      ))
    : allStudents.slice(0, 5); // Show first 5 by default
  
  const verificationInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Validation functions
  const hasUploadedImages = (card?: TrainingCard) => {
    const c = card || getActiveCard();
    return (c?.genuineFiles.length || 0) > 0 || (c?.forgedFiles.length || 0) > 0;
  };

  const canTrainModel = () => {
    // Enable when at least one card has student + images
    return cards.some(c => !!c.selectedStudent && hasUploadedImages(c));
  };

  // Data Quality Functions
  const getDataBalance = () => {
    const c = getActiveCard();
    const genuineCount = c?.genuineFiles.length || 0;
    const forgedCount = c?.forgedFiles.length || 0;
    const total = genuineCount + forgedCount;
    
    if (total === 0) return { ratio: 0, status: 'none', message: 'No data uploaded' };
    
    const genuineRatio = genuineCount / total;
    const forgedRatio = forgedCount / total;
    
    if (genuineRatio === 0 || forgedRatio === 0) {
      return { ratio: 0, status: 'unbalanced', message: 'Need both genuine and forged samples' };
    }
    
    const ratio = Math.min(genuineRatio, forgedRatio) / Math.max(genuineRatio, forgedRatio);
    
    if (ratio >= 0.7) {
      return { ratio, status: 'balanced', message: 'Good data balance' };
    } else if (ratio >= 0.4) {
      return { ratio, status: 'fair', message: 'Moderate imbalance - consider adding more samples' };
    } else {
      return { ratio, status: 'poor', message: 'Poor balance - add more samples of the minority class' };
    }
  };

  const getImageQualityScore = (file: File): number => {
    // Basic quality assessment based on file size only
    const sizeScore = Math.min(file.size / (1024 * 1024), 10) / 10; // Max 10MB = 1.0
    return sizeScore;
  };

  const getOverallDataQuality = () => {
    const balance = getDataBalance();
    const c = getActiveCard();
    const totalImages = (c?.genuineFiles.length || 0) + (c?.forgedFiles.length || 0);
    
    if (totalImages === 0) {
      return { score: 0, status: 'poor', message: 'No training data available' };
    }
    
    if (totalImages < 10) {
      return { score: 0.3, status: 'poor', message: 'Very few samples - recommend at least 10 images' };
    }
    
    if (totalImages < 20) {
      return { score: 0.6, status: 'fair', message: 'Limited samples - more data would improve accuracy' };
    }
    
    if (balance.status === 'balanced' && totalImages >= 20) {
      return { score: 0.9, status: 'excellent', message: 'Good training dataset' };
    }
    
    return { score: 0.7, status: 'good', message: 'Decent training dataset' };
  };

  const getTrainModelErrorMessage = () => {
    if (!selectedStudent && !hasUploadedImages()) {
      return "Please select a student and upload at least one signature image to train the model.";
    }
    if (!selectedStudent) {
      return "Please select a student to train the model.";
    }
    if (!hasUploadedImages()) {
      return "Please upload at least one signature image to train the model.";
    }
    return "";
  };

  const handleStudentSelection = (student: Student) => {
    const targetCardId = studentDialogForCardId || activeCardId;
    const card = cards.find(c => c.id === targetCardId) || getActiveCard();
    if (!card) return;

    const hasImgs = hasUploadedImages(card);
    const currentStudent = card.selectedStudent;
    if (hasImgs && currentStudent && currentStudent.id !== student.id) {
      setPendingStudent(student);
      setIsStudentSwitchConfirmOpen(true);
    } else {
      setCards(prev => prev.map(c => c.id === card.id ? { ...c, selectedStudent: student } : c));
      setSelectedStudent(student);
      setIsStudentDialogOpen(false);
      setStudentDialogForCardId(null);
    }
  };

  const confirmStudentSwitch = () => {
    const targetCard = cards.find(c => c.id === (studentDialogForCardId || activeCardId)) || getActiveCard();
    if (!targetCard) return;
    // Clear previews
    targetCard.genuineFiles.forEach(file => URL.revokeObjectURL(file.preview));
    targetCard.forgedFiles.forEach(file => URL.revokeObjectURL(file.preview));

    setCards(prev => prev.map(c => c.id === targetCard.id ? ({
      ...c,
      selectedStudent: pendingStudent,
      genuineFiles: [],
      forgedFiles: [],
    }) : c));

    setSelectedStudent(pendingStudent);
    setIsStudentDialogOpen(false);
    setIsStudentSwitchConfirmOpen(false);
    setPendingStudent(null);
    setStudentDialogForCardId(null);
    markDirty();
    toast({ title: "Student Changed", description: "All uploaded images have been cleared for the new student." });
  };

  const cancelStudentSwitch = () => {
    setIsStudentSwitchConfirmOpen(false);
    setPendingStudent(null);
  };

  // Training Functions
  const validateFiles = (files: File[]): File[] => {
    const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB to align with backend
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

  const handleTrainingFilesChange = async (files: File[], setType: 'genuine' | 'forged', cardId?: string) => {
    const targetId = cardId || activeCardId;
    const targetCard = cards.find(c => c.id === targetId) || getActiveCard();
    if (!targetCard) return;
    const safeFiles = validateFiles(files);
    const newFilesPromises = safeFiles.map(async (file) => {
      let preview = '';
      try {
        // Use native preview for most images; fallback to backend preview for TIFF or unknown
        if (file.type.startsWith('image/') && file.type !== 'image/tiff' && !file.name.toLowerCase().endsWith('.tif') && !file.name.toLowerCase().endsWith('.tiff')) {
          preview = URL.createObjectURL(file);
        } else {
          preview = await aiService.getPreviewURL(file);
        }
      } catch {
        preview = URL.createObjectURL(file);
      }
      return { file, preview };
    });
    const newFiles = await Promise.all(newFilesPromises);
    setCards(prev => prev.map(c => {
      if (c.id !== targetCard.id) return c;
      if (setType === 'genuine') {
        return { ...c, genuineFiles: [...c.genuineFiles, ...newFiles] };
      }
      return { ...c, forgedFiles: [...c.forgedFiles, ...newFiles] };
    }));
    markDirty();
  };

  const removeTrainingFile = (index: number, setType: 'genuine' | 'forged', cardId?: string) => {
    const targetId = cardId || activeCardId;
    setCards(prev => prev.map(c => {
      if (c.id !== targetId) return c;
      const list = setType === 'genuine' ? [...c.genuineFiles] : [...c.forgedFiles];
      if (list[index]) URL.revokeObjectURL(list[index].preview);
      list.splice(index, 1);
      return setType === 'genuine' ? { ...c, genuineFiles: list } : { ...c, forgedFiles: list };
    }));
    markDirty();
  };

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
    setTrainingLogs(['Training started...']);

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    const preparedCards = cards.filter(c => !!c.selectedStudent && ((c.genuineFiles.length + c.forgedFiles.length) > 0));
    const preparedIds = preparedCards.map(c => c.id);
    setCardProgress({});

    const computeGlobal = (map: Record<string, number>) => {
      if (preparedIds.length === 0) return 0;
      const total = preparedIds.reduce((sum, id) => sum + (map[id] || 0), 0);
      return total / preparedIds.length;
    };

    let index = 0;
    const trainNext = async () => {
      if (index >= preparedCards.length) {
        setGlobalProgress(100);
        setTrainingStage('completed');
        setTrainingStatus('All trainings completed');
        setIsTraining(false);
        return;
      }
      const card = preparedCards[index];
      setActiveCardId(card.id);
      setTrainingStatus(`Training ${index + 1} of ${preparedCards.length}`);

      try {
        const asyncResponse = await aiService.startAsyncTraining(
          card.selectedStudent!.student_id,
          card.genuineFiles.map(g => g.file),
          card.forgedFiles.map(f => f.file)
        );
        setJobId(asyncResponse.job_id);
        setTrainingLogs(prev => [...prev, `Job created: ${asyncResponse.job_id}`].slice(-15));

        const es = aiService.subscribeToJobProgress(
          asyncResponse.job_id,
          (job) => {
            if (job.current_stage) setTrainingStage(job.current_stage as any);
            setTrainingStatus(job.current_stage || `Training ${index + 1} of ${preparedCards.length}`);

            const newProgress = Math.max(0, Math.min(100, job.progress || 0));
            setCardProgress(prev => {
              const updated = { ...prev, [card.id]: Math.max(prev[card.id] || 0, newProgress) };
              setGlobalProgress(computeGlobal(updated));
              setTrainingProgress(computeGlobal(updated));
              return updated;
            });

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
                if (metrics.current_batch) logEntry += ` - Batch ${metrics.current_batch}`;
                logEntry += ` - Acc: ${(metrics.accuracy * 100).toFixed(1)}% - Loss: ${metrics.loss.toFixed(4)}`;
                if (metrics.val_accuracy > 0) {
                  logEntry += ` - Val Acc: ${(metrics.val_accuracy * 100).toFixed(1)}% - Val Loss: ${metrics.val_loss.toFixed(4)}`;
                }
                setTrainingLogs(prev => [...prev, logEntry].slice(-15));
                setCurrentEpochProgress({
                  epoch: metrics.current_epoch,
                  totalEpochs: metrics.total_epochs,
                  batch: metrics.current_batch || 0,
                  totalBatches: 0,
                  accuracy: metrics.accuracy,
                  loss: metrics.loss,
                  valAccuracy: metrics.val_accuracy || 0,
                  valLoss: metrics.val_loss || 0,
                });
              }
            }

            if (job.status === 'completed') {
              setTrainingResult(job.result);
              es.close();
              eventSourceRef.current = null;
              index += 1;
              setTimeout(trainNext, 0);
            } else if (job.status === 'failed') {
              setTrainingStage('error');
              setTrainingStatus('Training failed');
              toast({ title: "Training Failed", description: job.error || "Failed to complete training", variant: "destructive" });
              es.close();
              eventSourceRef.current = null;
              index += 1; // proceed to next to avoid blocking
              setTimeout(trainNext, 0);
            }
          },
          (error) => {
            console.error('Training progress error:', error);
            setTrainingStage('error');
            setTrainingStatus('Connection error');
            toast({ title: "Connection Error", description: "Lost connection to training progress updates", variant: "destructive" });
            es.close();
            eventSourceRef.current = null;
            index += 1;
            setTimeout(trainNext, 0);
          }
        );
        eventSourceRef.current = es;
      } catch (error) {
        console.error('Training error:', error);
        setTrainingStage('error');
        setTrainingStatus('Training failed');
        toast({ title: "Error", description: "An unexpected error occurred during training", variant: "destructive" });
        index += 1;
        setTimeout(trainNext, 0);
      }
    };

    trainNext();
  };

  // Verification Functions
  const handleVerificationFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.files?.[0];
    const valid = raw ? validateFiles([raw]) : [];
    const file = valid[0];
    if (file) {
      // Revoke previous preview URL if any
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

  // Reference files are optional UI-wise; if none provided, we'll fall back to
  // the currently uploaded Genuine training files for the selected student.

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
            
            // Stop camera stream
            const stream = video.srcObject as MediaStream;
            stream?.getTracks().forEach(track => track.stop());
            // Auto-trigger verification on capture
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

  const formatDuration = (ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Modal Functions
  type ModalContext = { kind: 'training', setType: 'genuine' | 'forged' } | { kind: 'verification' } | null;
  const [modalContext, setModalContext] = useState<ModalContext>(null);
  const getModalFilename = (): string => {
    if (!isModalOpen) return '';
    if (modalContext?.kind === 'training') {
      const card = getActiveCard();
      if (!card) return '';
      const files = modalContext.setType === 'genuine' ? card.genuineFiles : card.forgedFiles;
      const idx = files.findIndex(f => f.preview === modalImages[modalImageIndex]);
      return idx >= 0 ? files[idx].file.name : '';
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

  const removeAllTrainingFiles = (cardId?: string) => {
    const id = cardId || activeCardId;
    setCards(prev => prev.map(c => {
      if (c.id !== id) return c;
      c.genuineFiles.forEach(f => URL.revokeObjectURL(f.preview));
      c.forgedFiles.forEach(f => URL.revokeObjectURL(f.preview));
      return { ...c, genuineFiles: [], forgedFiles: [] };
    }));
    setIsDropdownOpen(false);
    toast({
      title: "Samples Removed",
      description: "All training samples have been removed",
    });
    markDirty();
  };

  const deleteModalCurrentImage = () => {
    if (!modalContext || modalContext.kind !== 'training') return;
    const card = getActiveCard();
    if (!card) return;
    const targetPreview = modalImages[modalImageIndex];
    if (modalContext.setType === 'genuine') {
      const idx = card.genuineFiles.findIndex(f => f.preview === targetPreview);
      if (idx !== -1) removeTrainingFile(idx, 'genuine', card.id);
      const updated = card.genuineFiles.filter(f => f.preview !== targetPreview).map(f => f.preview);
      setModalImages(updated);
    } else {
      const idx = card.forgedFiles.findIndex(f => f.preview === targetPreview);
      if (idx !== -1) removeTrainingFile(idx, 'forged', card.id);
      const updated = card.forgedFiles.filter(f => f.preview !== targetPreview).map(f => f.preview);
      setModalImages(updated);
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
      // Identification-based verify: only test_file
      const result = await aiService.verifySignature(verificationFile);
      setVerificationResult(result);
      
      if (result.success) {
        toast({
          title: "Verification Complete",
          description: result.is_unknown 
            ? "Signature not recognized - owner not trained" 
            : `Result: ${result.match ? 'Match found' : 'No match'}`,
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
          <h1 className="text-lg font-bold text-education-navy">SIGNATURE AI</h1>
          <p className="text-sm text-muted-foreground">
            Train AI models and verify signatures using machine learning
          </p>
        </div>

        {/* Merged Left Cards + Center Train + Right Verification */}
        <div ref={pageRef} className="relative">
          {/* Connectors Overlay */}
          <svg className="pointer-events-none absolute inset-0 z-0" xmlns="http://www.w3.org/2000/svg">
            {connectorPaths.map(p => (
              <path key={p.id} d={p.d} stroke="#94a3b8" strokeWidth="2" fill="none" />
            ))}
            {verifyConnector && (
              <path d={verifyConnector} stroke="#94a3b8" strokeWidth="2" fill="none" />
            )}
          </svg>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 relative z-10">
            {/* Left: Multiple merged student + training cards */}
            <div className="space-y-4">
              {cards.map((card) => (
                <Card key={card.id} className={`h-fit ${activeCardId === card.id ? 'ring-2 ring-education-navy' : ''}`} ref={(el) => { leftCardRefs.current[card.id] = el; }}>
                  <CardHeader className="py-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-left">
                        <span className="text-sm font-medium">
                          {card.selectedStudent ? `${card.selectedStudent.firstname} ${card.selectedStudent.surname}` : 'No student selected'}
                        </span>
                        {card.selectedStudent && (
                          <Badge variant="secondary" className="ml-2">{card.selectedStudent.student_id}</Badge>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Button variant="outline" size="sm" onClick={() => { setActiveCardId(card.id); setStudentDialogForCardId(card.id); setIsStudentDialogOpen(true); }}>
                          {card.selectedStudent ? 'Change' : 'Select'}
                        </Button>
                        <Button variant={activeCardId === card.id ? 'default' : 'outline'} size="icon" onClick={() => setActiveCardId(card.id)} aria-label="Make Active">
                          <User className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Upload buttons */}
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-2 hover:bg-transparent hover:text-foreground"
                        onClick={() => {
                          setActiveCardId(card.id);
                          setCurrentTrainingSet('forged');
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
                        Forged ({card.forgedFiles.length})
                      </Button>
                      <Button
                        variant="default"
                        size="sm"
                        className="flex items-center gap-2"
                        onClick={() => {
                          setActiveCardId(card.id);
                          setCurrentTrainingSet('genuine');
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
                        Genuine ({card.genuineFiles.length})
                      </Button>
                    </div>

                    {/* Preview grid */}
                    <div className="relative w-full h-48 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50">
                      {(card.genuineFiles.length + card.forgedFiles.length) > 0 ? (
                        <div className="grid grid-cols-3 gap-2 w-full h-full p-3 overflow-y-auto">
                          {(currentTrainingSet === 'genuine' ? card.genuineFiles : card.forgedFiles).slice(0, 60).map((item, index) => (
                            <div key={index} className="relative group/itm cursor-pointer">
                              <img src={item.preview} alt={`Sample ${index+1}`} className="w-full h-16 object-cover rounded border hover:opacity-80 transition-opacity" />
                              <Button size="sm" variant="destructive" className="absolute top-1 right-1 w-6 h-6 p-0 opacity-0 group-hover/itm:opacity-100 transition-opacity flex items-center justify-center"
                                onClick={(e) => { e.stopPropagation(); removeTrainingFile(index, currentTrainingSet, card.id); }}>
                                <Trash2 className="w-3.5 h-3.5" />
                              </Button>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center text-gray-500">
                          <Upload className="w-8 h-8 mx-auto mb-2" />
                          <p>No images uploaded</p>
                        </div>
                      )}
                    </div>

                    {/* Add another card + remove samples */}
                    <div className="flex items-center justify-between">
                      <Button variant="secondary" size="sm" onClick={() => removeAllTrainingFiles(card.id)} disabled={(card.genuineFiles.length + card.forgedFiles.length) === 0}>Remove All</Button>
                      <Button variant="outline" size="sm" onClick={() => setCards(prev => [...prev, { id: crypto.randomUUID(), selectedStudent: null, genuineFiles: [], forgedFiles: [] }])}>Add</Button>
                    </div>
                    {typeof cardProgress[card.id] === 'number' && (
                      <div className="text-xs text-muted-foreground">Progress: {Math.round(cardProgress[card.id])}%</div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Center: Train Model button */}
            <div className="flex items-start justify-center">
              <div className="w-full max-w-sm">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><Brain className="w-5 h-5" />Train Model</CardTitle>
                    <CardDescription>Train all prepared student cards together</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button ref={centerButtonRef} onClick={handleTrainModel} disabled={!canTrainModel() || isTraining} className="w-full">
                      {isTraining ? (<><Loader2 className="w-4 h-4 mr-2 animate-spin" />Training {Math.round(globalProgress)}%</>) : (<><Brain className="w-4 h-4 mr-2" />Train Model</>)}
                    </Button>
                    {isTraining && (
                      <div className="text-xs text-muted-foreground">Overall progress: {Math.round(globalProgress)}%</div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Right: Verification */}
            <div ref={verificationCardRef}>
              {/* Signature Verification Section */}
              <Card className="h-fit">
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
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0 opacity-60 text-muted-foreground hover:opacity-100 hover:text-foreground hover:bg-transparent transition-colors"
                        >
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={() => {
                            setVerificationFile(null);
                            setVerificationPreview('');
                            stopCamera();
                          }}
                          disabled={!verificationFile && !useCamera && !verificationPreview}
                        >
                          Clear
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
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
                          <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover rounded-lg" />
                        </div>
                      ) : verificationPreview ? (
                        <img src={verificationPreview} alt="Signature preview" className="w-full h-full object-contain rounded-lg cursor-pointer hover:opacity-80 transition-opacity" onClick={() => openImageModal([verificationPreview], 0)} />
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
                      <Button onClick={capturePhoto} size="sm" className="flex-1">Capture</Button>
                      <Button onClick={stopCamera} variant="outline" size="sm" className="flex-1">Cancel</Button>
                    </div>
                  )}
                  {/* Hidden File Input */}
                  <Input ref={verificationInputRef} type="file" accept="image/*" onChange={handleVerificationFileChange} className="hidden" />
                  {/* Verify Button */}
                  <Button onClick={handleVerifySignature} disabled={!verificationFile || isVerifying} className="w-full">
                    {isVerifying ? (<><Loader2 className="w-4 h-4 mr-2 animate-spin" />Verifying...</>) : (<><Scan className="w-4 h-4 mr-2" />Verify Signature</>)}
                  </Button>
                  {/* Verification Result */}
                  {verificationResult && (
                    <Alert className={
                      verificationResult.is_unknown 
                        ? "border-orange-200 bg-orange-50" 
                        : verificationResult.match 
                          ? "border-green-200 bg-green-50" 
                          : "border-red-200 bg-red-50"
                    }>
                      <div className="flex items-center gap-2">
                        {verificationResult.is_unknown ? (
                          <AlertTriangle className="w-4 h-4 text-orange-600" />
                        ) : verificationResult.match ? (
                          <CheckCircle className="w-4 h-4 text-green-600" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-600" />
                        )}
                        <AlertDescription>
                          <div className="space-y-2">
                            <p>
                              <strong>Result:</strong> {
                                verificationResult.is_unknown 
                                  ? 'Signature Not Recognized' 
                                  : verificationResult.match 
                                    ? 'Match Found' 
                                    : 'No Match'
                              }
                            </p>
                            <p>
                              <strong>Confidence:</strong> {
                                verificationResult.is_unknown 
                                  ? 'Low (Unknown Signature)' 
                                  : `${(verificationResult.score * 100).toFixed(1)}%`
                              }
                            </p>
                            {verificationResult.predicted_student && !verificationResult.is_unknown && (
                              <p>
                                <strong>Predicted Student:</strong> {verificationResult.predicted_student.firstname} {verificationResult.predicted_student.surname}
                              </p>
                            )}
                            {verificationResult.is_unknown && (
                              <p className="text-orange-700 font-medium">
                                <strong>Status:</strong> Owner not trained or signature too different from trained samples
                              </p>
                            )}
                            <p className="text-sm text-muted-foreground">
                              {verificationResult.message}
                            </p>
                          </div>
                        </AlertDescription>
                      </div>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
            {/* START REMOVE LEGACY */}
            {/* END REMOVE LEGACY */}
            {/* LEGACY BLOCK REMOVED
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Model Training
                  </CardTitle>
                  <CardDescription>
                    Upload signature samples to train AI model for a specific student
                  </CardDescription>
                </div>
                <DropdownMenu open={isDropdownOpen} onOpenChange={setIsDropdownOpen}>
                  <DropdownMenuTrigger asChild>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      className="h-6 w-6 p-0 opacity-60 text-muted-foreground hover:opacity-100 hover:text-foreground hover:bg-transparent transition-colors"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem 
                      onClick={() => setIsConfirmRemoveOpen(true)} 
                      className="text-red-600"
                      disabled={(genuineFiles.length + forgedFiles.length) === 0}
                    >
                      Remove All Samples
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Upload Buttons: Forged (left, normal) and Genuine (right, highlighted) */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-2 hover:bg-transparent hover:text-foreground"
                  onClick={() => {
                    setCurrentTrainingSet('forged');
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'image/*';
                    input.multiple = true;
                    input.onchange = (e) => {
                      const files = Array.from((e.target as HTMLInputElement).files || []);
                      handleTrainingFilesChange(files, 'forged');
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
                    setCurrentTrainingSet('genuine');
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'image/*';
                    input.multiple = true;
                    input.onchange = (e) => {
                      const files = Array.from((e.target as HTMLInputElement).files || []);
                      handleTrainingFilesChange(files, 'genuine');
                    };
                    input.click();
                  }}
                >
                  <Upload className="w-4 h-4" />
                  Genuine
                </Button>
              </div>

              {/* Large Square Preview Box for Training Images (Genuine/Forged switch) */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Training Images Preview</Label>
                  <div className="text-xs text-muted-foreground">
                    {currentTrainingSet === 'genuine' ? (
                      <span>Genuine</span>
                    ) : (
                      <span>Forged</span>
                    )}
                  </div>
                </div>
                <div className="relative w-full h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center bg-gray-50 group">
                  {/* Hover Previous/Next inside box */}
                  <button
                    className="hidden group-hover:flex absolute left-2 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white rounded-full p-2 shadow"
                    onClick={() => setCurrentTrainingSet(prev => prev === 'genuine' ? 'forged' : 'genuine')}
                    aria-label="Previous"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <button
                    className="hidden group-hover:flex absolute right-2 top-1/2 -translate-y-1/2 bg-white/80 hover:bg-white rounded-full p-2 shadow"
                    onClick={() => setCurrentTrainingSet(prev => prev === 'genuine' ? 'forged' : 'genuine')}
                    aria-label="Next"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>

                  {/* This legacy preview block is removed in favor of per-card previews above */}
                  {false ? (
                    <div className="grid grid-cols-3 gap-2 w-full h-full p-4 overflow-y-auto">
                      {[]
                        .slice(0, visibleCounts[currentTrainingSet] || 0)
                        .map((item, index) => (
                        <div key={index} className="relative group/itm cursor-pointer" onClick={() => openImageModal((currentTrainingSet === 'genuine' ? genuineFiles : forgedFiles).map(f => f.preview), index, { kind: 'training', setType: currentTrainingSet })}>
                          <img
                            src={''}
                            alt={`Sample ${index + 1}`}
                            className="w-full h-16 object-cover rounded border hover:opacity-80 transition-opacity"
                            loading="lazy"
                          />
                          <Button
                            size="sm"
                            variant="destructive"
                            className="absolute top-1 right-1 w-6 h-6 p-0 opacity-0 group-hover/itm:opacity-100 transition-opacity flex items-center justify-center"
                            onClick={(e) => {
                              e.stopPropagation();
                              // legacy
                            }}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        </div>
                      ))}
                      {false && (
                        <div className="col-span-3 flex justify-center pt-2">
                          <button
                            className="text-xs text-foreground hover:underline"
                            onClick={() =>
                              setVisibleCounts((prev) => ({
                                ...prev,
                                [currentTrainingSet]: prev[currentTrainingSet] + 60,
                              }))
                            }
                          >
                            Show more
                          </button>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center text-gray-500">
                      <Upload className="w-8 h-8 mx-auto mb-2" />
                      <p>No images uploaded</p>
                    </div>
                  )}
                </div>
              </div>





              {/* Train Button */}
              <Button
                onClick={handleTrainModel}
                disabled={!canTrainModel() || isTraining}
                className="w-full"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    {trainingStatus} {Math.round(trainingProgress)}%
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 mr-2" />
                    Train Model
                  </>
                )}
              </Button>

              {isTraining && (
                <div className="space-y-3">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Progress: {Math.round(trainingProgress)}%</span>
                    <span>{estimatedTimeRemaining || ''}</span>
                  </div>
                  
                  {/* Real-time Training Logs Display */}
                  {trainingLogs.length > 0 && (
                    <div className="bg-gray-900 text-green-400 rounded-lg p-3 font-mono text-xs max-h-64 overflow-y-auto">
                      <div className="text-green-300 text-sm font-semibold mb-2">Training Progress (Live)</div>
                      {trainingLogs.map((log, index) => (
                        <div key={index} className="mb-1">
                          {log}
                        </div>
                      ))}
                      {currentEpochProgress && (
                        <div className="mt-2 pt-2 border-t border-gray-700">
                          <div className="text-yellow-400">
                            Current: Epoch {currentEpochProgress.epoch}/{currentEpochProgress.totalEpochs}
                            {currentEpochProgress.batch > 0 && ` - Batch ${currentEpochProgress.batch}`} - 
                            Acc: {(currentEpochProgress.accuracy * 100).toFixed(1)}% - 
                            Loss: {currentEpochProgress.loss.toFixed(4)}
                            {currentEpochProgress.valAccuracy > 0 && (
                              <> - Val Acc: {(currentEpochProgress.valAccuracy * 100).toFixed(1)}% - Val Loss: {currentEpochProgress.valLoss.toFixed(4)}</>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Training Results */}
              {trainingResult && (
                <div className="space-y-3">
                  <Alert className={trainingResult.success ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}>
                    <div className="flex items-center gap-2">
                      {trainingResult.success ? (
                        <CheckCircle className="w-4 h-4 text-green-600" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-red-600" />
                      )}
                      <AlertDescription>
                        <div className="space-y-2">
                          <p>
                            <strong>Status:</strong> {trainingResult.success ? 'Training Completed' : 'Training Failed'}
                          </p>
                          {trainingResult.profile && (
                            <>
                              <p>
                                <strong>Model Status:</strong>{' '}
                                <Badge variant={trainingResult.profile.status === 'ready' ? 'default' : 'secondary'}>
                                  {trainingResult.profile.status}
                                </Badge>
                              </p>
                              <p>
                                <strong>Samples:</strong> {trainingResult.profile.num_samples}
                              </p>
                              {trainingResult.profile.last_trained_at && (
                                <p>
                                  <strong>Last Trained:</strong>{' '}
                                  {new Date(trainingResult.profile.last_trained_at).toLocaleString()}
                                </p>
                              )}
                            </>
                          )}
                          <p className="text-sm text-muted-foreground">
                            {trainingResult.message}
                          </p>
                        </div>
                    </AlertDescription>
                  </div>
                </Alert>
                
                {/* Training Metrics */}
                {trainingResult.success && (
                  <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <h4 className="text-sm font-medium text-green-900 mb-3">Training Metrics</h4>
                    <div className="grid grid-cols-2 gap-4 text-xs">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-green-700">Train Acc:</span>
                          <span className="font-medium text-green-800">{typeof trainingResult.accuracy === 'number' ? trainingResult.accuracy.toFixed(3) : '—'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-green-700">Val Acc:</span>
                          <span className="font-medium text-green-800">{typeof trainingResult.val_accuracy === 'number' ? trainingResult.val_accuracy.toFixed(3) : '—'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-green-700">Precision:</span>
                          <span className="font-medium text-green-800">{typeof trainingResult.precision === 'number' ? trainingResult.precision.toFixed(3) : '—'}</span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-green-700">Recall:</span>
                          <span className="font-medium text-green-800">{typeof trainingResult.recall === 'number' ? trainingResult.recall.toFixed(3) : '—'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-green-700">F1 Score:</span>
                          <span className="font-medium text-green-800">{typeof trainingResult.f1 === 'number' ? trainingResult.f1.toFixed(3) : '—'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-green-700">Training Time:</span>
                          <span className="font-medium text-green-800">{typeof trainingResult.train_time_s === 'number' ? `${trainingResult.train_time_s}s` : '—'}</span>
                        </div>
                      </div>
                    </div>
                    <div className="mt-3 text-xs text-green-700">
                      Model is ready for signature verification!
                      {trainingResult.calibration && (
                        <div className="mt-2 grid grid-cols-3 gap-2">
                          <div><strong>Thr:</strong> {typeof trainingResult.calibration.threshold === 'number' ? trainingResult.calibration.threshold.toFixed(4) : '—'}</div>
                          <div><strong>FAR:</strong> {typeof trainingResult.calibration.far === 'number' ? trainingResult.calibration.far.toFixed(3) : '—'}</div>
                          <div><strong>FRR:</strong> {typeof trainingResult.calibration.frr === 'number' ? trainingResult.calibration.frr.toFixed(3) : '—'}</div>
                        </div>
                      )}
                      <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded text-blue-800">
                        <strong>Data Augmentation Applied:</strong> Rotation, scale, brightness, blur, and thickness variations for improved robustness
                      </div>
                    </div>
                  </div>
                )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Signature Verification Section (legacy duplicate) */}
          <Card className="h-fit">
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
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-6 p-0 opacity-60 text-muted-foreground hover:opacity-100 hover:text-foreground hover:bg-transparent transition-colors"
                    >
                      <MoreVertical className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      onClick={() => {
                        setVerificationFile(null);
                        setVerificationPreview('');
                        stopCamera();
                      }}
                      disabled={!verificationFile && !useCamera && !verificationPreview}
                    >
                      Clear
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
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

              {/* Reference images are optional now; if not provided,
                 Genuine training uploads will be used for verification */}

              {/* Verification Result */}
              {verificationResult && (
                <Alert className={
                  verificationResult.is_unknown 
                    ? "border-orange-200 bg-orange-50" 
                    : verificationResult.match 
                      ? "border-green-200 bg-green-50" 
                      : "border-red-200 bg-red-50"
                }>
                  <div className="flex items-center gap-2">
                    {verificationResult.is_unknown ? (
                      <AlertTriangle className="w-4 h-4 text-orange-600" />
                    ) : verificationResult.match ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : (
                      <XCircle className="w-4 h-4 text-red-600" />
                    )}
                    <AlertDescription>
                      <div className="space-y-2">
                        <p>
                          <strong>Result:</strong> {
                            verificationResult.is_unknown 
                              ? 'Signature Not Recognized' 
                              : verificationResult.match 
                                ? 'Match Found' 
                                : 'No Match'
                          }
                        </p>
                        <p>
                          <strong>Confidence:</strong> {
                            verificationResult.is_unknown 
                              ? 'Low (Unknown Signature)' 
                              : `${(verificationResult.score * 100).toFixed(1)}%`
                          }
                        </p>
                        {verificationResult.predicted_student && !verificationResult.is_unknown && (
                          <p>
                            <strong>Predicted Student:</strong> {verificationResult.predicted_student.firstname} {verificationResult.predicted_student.surname}
                          </p>
                        )}
                        {verificationResult.is_unknown && (
                          <p className="text-orange-700 font-medium">
                            <strong>Status:</strong> Owner not trained or signature too different from trained samples
                          </p>
                        )}
                        <p className="text-sm text-muted-foreground">
                          {verificationResult.message}
                        </p>
                        
                        {/* Anti-Spoofing Warnings */}
                        {verificationResult.antispoofing && (
                          <div className="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                            <div className="flex items-start gap-2">
                              <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                              <div className="space-y-2 text-sm">
                                <p className="font-medium text-amber-800">
                                  {verificationResult.antispoofing.warning_message}
                                </p>
                                {verificationResult.antispoofing.is_potentially_spoofed && (
                                  <div className="grid grid-cols-2 gap-2 text-xs text-amber-700">
                                    {verificationResult.antispoofing.is_likely_printed && (
                                      <div>
                                        <strong>Printed:</strong> {(verificationResult.antispoofing.printed_confidence * 100).toFixed(1)}%
                                      </div>
                                    )}
                                    {verificationResult.antispoofing.is_low_quality && (
                                      <div>
                                        <strong>Quality:</strong> {(verificationResult.antispoofing.quality_score * 100).toFixed(1)}%
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </AlertDescription>
                  </div>
                </Alert>
              )}
            </CardContent>
          </Card>
        </div>
            */}
        {/* Image Preview Modal */
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
                    {/* Prev/Next Arrows - inside image boundary, show on hover */}
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

                  {/* Count + Trash inline with translucent background (old style) */}
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
        <Dialog open={isStudentDialogOpen} onOpenChange={(v) => { setIsStudentDialogOpen(v); if (!v) setStudentDialogForCardId(null);} }>
          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle>Select Student</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <Input
                placeholder="Search by ID or Name"
                value={studentSearch}
                onChange={(e) => setStudentSearch(e.target.value)}
              />
              <div className="max-h-64 overflow-auto border rounded-md">
                {isLoadingStudents ? (
                  <div className="p-4 text-sm text-muted-foreground">Loading students…</div>
                ) : isStudentSearching ? (
                  <div className="p-4 text-sm text-muted-foreground">Searching…</div>
                ) : filteredStudents.length === 0 ? (
                  <div className="p-4 text-sm text-muted-foreground">No results</div>
                ) : (
                  <ul className="divide-y">
                    {filteredStudents.map((s) => (
                      <li key={s.id}>
                        <button
                          className="w-full text-left p-3 hover:bg-muted/50"
                          onClick={() => handleStudentSelection(s)}
                        >
                          <div className="font-medium">{`${s.firstname} ${s.surname}`}</div>
                          <div className="text-xs text-muted-foreground">ID: {s.student_id} • {s.program} • Year {s.year} • Sec {s.section}</div>
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </DialogContent>
        </Dialog>
        {/* Confirm Remove All Samples */}
        <Dialog open={isConfirmRemoveOpen} onOpenChange={setIsConfirmRemoveOpen}>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle>Remove all samples?</DialogTitle>
            </DialogHeader>
            <div className="text-sm text-muted-foreground">
              This action will remove all Genuine and Forged samples. This cannot be undone.
            </div>
            <div className="flex justify-end gap-2 pt-4">
              <Button variant="outline" size="sm" onClick={() => setIsConfirmRemoveOpen(false)}>Cancel</Button>
              <Button variant="destructive" size="sm" onClick={() => { removeAllTrainingFiles(); setIsConfirmRemoveOpen(false); }}>Remove</Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Student Switch Confirmation Dialog */}
        <Dialog open={isStudentSwitchConfirmOpen} onOpenChange={setIsStudentSwitchConfirmOpen}>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle>Change Student?</DialogTitle>
            </DialogHeader>
            <div className="text-sm text-muted-foreground mb-4">
              You have uploaded signature images for the current student. Changing the student will automatically clear all uploaded images.
            </div>
            <div className="flex gap-2 justify-end">
              <Button variant="outline" onClick={cancelStudentSwitch}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={confirmStudentSwitch}>
                Change Student
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        {/* Unsaved Changes Dialog */}
        <UnsavedChangesDialog
          open={showConfirmDialog}
          onConfirm={() => {
            confirmClose();
            // Clear any pending navigation
            setPendingNavigation(null);
          }}
          onCancel={() => {
            cancelClose();
            // Clear any pending navigation
            setPendingNavigation(null);
          }}
        />
      </div>
    </Layout>
  );
};

export default SignatureAI;