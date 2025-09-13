import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { 
  Upload, 
  Brain, 
  ChevronDown,
  MoreVertical,
  Trash2,
  User,
  X,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import type { Student, StudentTrainingCard as StudentTrainingCardType, TrainingFile } from '@/types';

interface StudentTrainingCardProps {
  card: StudentTrainingCardType;
  index: number;
  onUpdate: (updates: Partial<StudentTrainingCardType>) => void;
  onRemove: () => void;
  onOpenStudentDialog: () => void;
  onTrainingFilesChange: (files: File[], setType: 'genuine' | 'forged', cardId: string) => void;
  onRemoveTrainingFile: (index: number, setType: 'genuine' | 'forged', cardId: string) => void;
  onOpenImageModal: (images: string[], startIndex: number, context: { kind: 'training', setType: 'genuine' | 'forged', cardId: string } | { kind: 'verification' } | null) => void;
  onRemoveAllSamples: (cardId: string) => void;
  onCardClick: () => void;
  locked?: boolean;
}

const StudentTrainingCard: React.FC<StudentTrainingCardProps> = ({
  card,
  index,
  onUpdate,
  onRemove,
  onOpenStudentDialog,
  onTrainingFilesChange,
  onRemoveTrainingFile,
  onOpenImageModal,
  onRemoveAllSamples,
  onCardClick,
  locked
}) => {
  const hasUploadedImages = () => {
    return card.genuineFiles.length > 0 || card.forgedFiles.length > 0;
  };

  const handleCardClick = (e: React.MouseEvent) => {
    // Ignore clicks while confirmation dialog is open
    if (isRemoveConfirmOpen) return;
    // Ignore clicks originating from any dialog/portal elements
    const target = e.target as HTMLElement;
    if (
      target.closest('[role="dialog"]') ||
      target.closest('[data-radix-portal]')
    ) {
      return;
    }
    onCardClick();
  };

  const [isRemoveConfirmOpen, setIsRemoveConfirmOpen] = useState(false);

  return (
    <Card 
      className="relative cursor-pointer hover:shadow-md transition-shadow h-10 group"
      onClick={handleCardClick}
    >
      <CardContent className="py-0 px-2 pr-1 h-full flex flex-col justify-center">
        {/* Header with student info, counts, and remove */}
        <div className="flex items-center justify-between">
          <div className="min-w-0 flex-1 text-xs">
            <span className="font-medium truncate">
              {card.student ? `${card.student.firstname} ${card.student.surname}` : 'No student selected'}
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs mr-1">
            <span className="text-green-600 font-medium">{card.genuineCount ?? card.genuineFiles.length}</span>
            <span className="text-red-600 font-medium">{card.forgedCount ?? card.forgedFiles.length}</span>
            {(card.genuineFiles.some(f => f.placeholder) || card.forgedFiles.some(f => f.placeholder)) && (
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" title="Loading images..." />
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-5 w-5 p-0 opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground hover:bg-transparent transition-opacity shrink-0"
            disabled={locked}
            onClick={(e) => {
              e.stopPropagation();
              if (locked) return;
              setIsRemoveConfirmOpen(true);
            }}
            aria-label="Remove Student"
          >
            <X className="h-3 w-3" />
          </Button>
        </div>
      </CardContent>

      {/* Confirm Remove Dialog */}
      <Dialog open={isRemoveConfirmOpen} onOpenChange={setIsRemoveConfirmOpen}>
        <DialogContent className="max-w-sm" onClick={(e) => e.stopPropagation()}>
          <DialogHeader>
            <DialogTitle>Remove Student</DialogTitle>
          </DialogHeader>
          <div className="text-sm">Are you sure you want to remove this student?</div>
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" size="sm" onClick={(e) => { e.stopPropagation(); setIsRemoveConfirmOpen(false); }}>Cancel</Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                setIsRemoveConfirmOpen(false);
                onRemove();
              }}
            >
              Yes
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
};

export default StudentTrainingCard;
