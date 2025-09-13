import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import StudentTrainingCard from '../StudentTrainingCard';
import type { Student, StudentTrainingCard as StudentTrainingCardType } from '@/types';

const mockStudent: Student = {
  id: 1,
  student_id: 'STU001',
  firstname: 'John',
  surname: 'Doe',
  program: 'Computer Science',
  year: '1st',
  section: 'A',
  created_at: '2023-01-01T00:00:00Z',
  updated_at: '2023-01-01T00:00:00Z'
};

const mockCard: StudentTrainingCardType = {
  id: 'card-1',
  student: mockStudent,
  genuineFiles: [],
  forgedFiles: [],
  isExpanded: true,
  genuineCount: 5,
  forgedCount: 2
};

const mockCardWithPlaceholders: StudentTrainingCardType = {
  id: 'card-2',
  student: mockStudent,
  genuineFiles: [
    {
      file: new File(['test'], 'placeholder1.jpg'),
      preview: '',
      placeholder: true,
      label: 'genuine'
    },
    {
      file: new File(['test'], 'placeholder2.jpg'),
      preview: '',
      placeholder: true,
      label: 'genuine'
    }
  ],
  forgedFiles: [
    {
      file: new File(['test'], 'placeholder3.jpg'),
      preview: '',
      placeholder: true,
      label: 'forged'
    }
  ],
  isExpanded: true,
  genuineCount: 2,
  forgedCount: 1
};

const mockCardWithImages: StudentTrainingCardType = {
  id: 'card-3',
  student: mockStudent,
  genuineFiles: [
    {
      file: new File(['test'], 'genuine1.jpg'),
      preview: 'https://s3.amazonaws.com/bucket/genuine1.jpg',
      id: 1,
      s3Key: 'signatures/1/genuine/genuine1.jpg',
      label: 'genuine'
    }
  ],
  forgedFiles: [
    {
      file: new File(['test'], 'forged1.jpg'),
      preview: 'https://s3.amazonaws.com/bucket/forged1.jpg',
      id: 2,
      s3Key: 'signatures/1/forged/forged1.jpg',
      label: 'forged'
    }
  ],
  isExpanded: true,
  genuineCount: 1,
  forgedCount: 1
};

const defaultProps = {
  card: mockCard,
  index: 0,
  onUpdate: jest.fn(),
  onRemove: jest.fn(),
  onOpenStudentDialog: jest.fn(),
  onTrainingFilesChange: jest.fn(),
  onRemoveTrainingFile: jest.fn(),
  onOpenImageModal: jest.fn(),
  onRemoveAllSamples: jest.fn(),
  onCardClick: jest.fn(),
  locked: false
};

describe('StudentTrainingCard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render student information correctly', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument(); // genuine count
    expect(screen.getByText('2')).toBeInTheDocument(); // forged count
  });

  it('should render "No student selected" when no student is assigned', () => {
    const cardWithoutStudent = { ...mockCard, student: null };
    render(<StudentTrainingCard {...defaultProps} card={cardWithoutStudent} />);

    expect(screen.getByText('No student selected')).toBeInTheDocument();
  });

  it('should show loading indicator when placeholder files are present', () => {
    render(<StudentTrainingCard {...defaultProps} card={mockCardWithPlaceholders} />);

    // Should show the loading indicator (blue dot)
    const loadingIndicator = screen.getByTitle('Loading images...');
    expect(loadingIndicator).toBeInTheDocument();
  });

  it('should not show loading indicator when no placeholder files are present', () => {
    render(<StudentTrainingCard {...defaultProps} card={mockCardWithImages} />);

    // Should not show the loading indicator
    expect(screen.queryByTitle('Loading images...')).not.toBeInTheDocument();
  });

  it('should call onCardClick when card is clicked', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    const card = screen.getByText('John Doe').closest('.cursor-pointer');
    fireEvent.click(card!);

    expect(defaultProps.onCardClick).toHaveBeenCalled();
  });

  it('should not call onCardClick when remove confirmation dialog is open', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    // Click the remove button to open confirmation dialog
    const removeButton = screen.getByLabelText('Remove Student');
    fireEvent.click(removeButton);

    // Click the card while dialog is open
    const card = screen.getByText('John Doe').closest('.cursor-pointer');
    fireEvent.click(card!);

    expect(defaultProps.onCardClick).not.toHaveBeenCalled();
  });

  it('should show remove button on hover', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    const card = screen.getByText('John Doe').closest('.group');
    fireEvent.mouseEnter(card!);

    const removeButton = screen.getByLabelText('Remove Student');
    expect(removeButton).toBeInTheDocument();
  });

  it('should open remove confirmation dialog when remove button is clicked', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    const removeButton = screen.getByLabelText('Remove Student');
    fireEvent.click(removeButton);

    expect(screen.getByText('Remove Student')).toBeInTheDocument();
    expect(screen.getByText('Are you sure you want to remove this student?')).toBeInTheDocument();
  });

  it('should call onRemove when confirmation is accepted', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    // Click the remove button to open confirmation dialog
    const removeButton = screen.getByLabelText('Remove Student');
    fireEvent.click(removeButton);

    // Click "Yes" to confirm removal
    const confirmButton = screen.getByText('Yes');
    fireEvent.click(confirmButton);

    expect(defaultProps.onRemove).toHaveBeenCalled();
  });

  it('should close confirmation dialog when "Cancel" is clicked', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    // Click the remove button to open confirmation dialog
    const removeButton = screen.getByLabelText('Remove Student');
    fireEvent.click(removeButton);

    // Click "Cancel" to close dialog
    const cancelButton = screen.getByText('Cancel');
    fireEvent.click(cancelButton);

    expect(screen.queryByText('Are you sure you want to remove this student?')).not.toBeInTheDocument();
  });

  it('should disable remove button when locked', () => {
    render(<StudentTrainingCard {...defaultProps} locked={true} />);

    const removeButton = screen.getByLabelText('Remove Student');
    expect(removeButton).toBeDisabled();
  });

  it('should not call onRemove when locked and remove button is clicked', () => {
    render(<StudentTrainingCard {...defaultProps} locked={true} />);

    const removeButton = screen.getByLabelText('Remove Student');
    fireEvent.click(removeButton);

    // Should not open confirmation dialog when locked
    expect(screen.queryByText('Are you sure you want to remove this student?')).not.toBeInTheDocument();
    expect(defaultProps.onRemove).not.toHaveBeenCalled();
  });

  it('should prevent event propagation when remove button is clicked', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    const removeButton = screen.getByLabelText('Remove Student');
    const card = screen.getByText('John Doe').closest('.cursor-pointer');

    // Mock stopPropagation
    const stopPropagation = jest.fn();
    fireEvent.click(removeButton, { stopPropagation });

    expect(stopPropagation).toHaveBeenCalled();
    expect(defaultProps.onCardClick).not.toHaveBeenCalled();
  });

  it('should prevent event propagation when confirmation dialog is clicked', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    // Click the remove button to open confirmation dialog
    const removeButton = screen.getByLabelText('Remove Student');
    fireEvent.click(removeButton);

    const dialog = screen.getByText('Are you sure you want to remove this student?').closest('[role="dialog"]');
    const stopPropagation = jest.fn();
    fireEvent.click(dialog!, { stopPropagation });

    expect(stopPropagation).toHaveBeenCalled();
  });

  it('should handle card click with dialog elements', () => {
    render(<StudentTrainingCard {...defaultProps} />);

    // Click the remove button to open confirmation dialog
    const removeButton = screen.getByLabelText('Remove Student');
    fireEvent.click(removeButton);

    // Click on a dialog element (should not trigger card click)
    const dialog = screen.getByText('Are you sure you want to remove this student?').closest('[role="dialog"]');
    fireEvent.click(dialog!);

    expect(defaultProps.onCardClick).not.toHaveBeenCalled();
  });

  it('should display correct counts for genuine and forged files', () => {
    const cardWithCounts = {
      ...mockCard,
      genuineCount: 10,
      forgedCount: 5
    };

    render(<StudentTrainingCard {...defaultProps} card={cardWithCounts} />);

    expect(screen.getByText('10')).toBeInTheDocument(); // genuine count
    expect(screen.getByText('5')).toBeInTheDocument(); // forged count
  });

  it('should fall back to file array length when counts are not provided', () => {
    const cardWithoutCounts = {
      ...mockCard,
      genuineCount: undefined,
      forgedCount: undefined,
      genuineFiles: [
        { file: new File(['test'], 'genuine1.jpg'), preview: 'url1', label: 'genuine' as const },
        { file: new File(['test'], 'genuine2.jpg'), preview: 'url2', label: 'genuine' as const }
      ],
      forgedFiles: [
        { file: new File(['test'], 'forged1.jpg'), preview: 'url3', label: 'forged' as const }
      ]
    };

    render(<StudentTrainingCard {...defaultProps} card={cardWithoutCounts} />);

    expect(screen.getByText('2')).toBeInTheDocument(); // genuine files length
    expect(screen.getByText('1')).toBeInTheDocument(); // forged files length
  });
});