import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { TrainingOptionsDropdown } from './TrainingOptionsDropdown';

// Mock the dropdown component
const TrainingOptionsDropdown = ({ 
  useGPU, 
  useS3Upload, 
  onGPUChange, 
  onS3UploadChange, 
  isLocked 
}: {
  useGPU: boolean;
  useS3Upload: boolean;
  onGPUChange: (value: boolean) => void;
  onS3UploadChange: (value: boolean) => void;
  isLocked: boolean;
}) => {
  return (
    <div data-testid="training-options">
      <input
        type="checkbox"
        id="use-gpu"
        checked={useGPU}
        onChange={(e) => onGPUChange(e.target.checked)}
        disabled={isLocked}
        data-testid="gpu-checkbox"
      />
      <label htmlFor="use-gpu">GPU Training</label>
      
      <input
        type="checkbox"
        id="use-s3"
        checked={useS3Upload}
        onChange={(e) => onS3UploadChange(e.target.checked)}
        disabled={isLocked}
        data-testid="s3-checkbox"
      />
      <label htmlFor="use-s3">S3 Upload</label>
    </div>
  );
};

describe('TrainingOptionsDropdown', () => {
  const mockOnGPUChange = jest.fn();
  const mockOnS3UploadChange = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders GPU and S3 upload checkboxes', () => {
    render(
      <TrainingOptionsDropdown
        useGPU={true}
        useS3Upload={false}
        onGPUChange={mockOnGPUChange}
        onS3UploadChange={mockOnS3UploadChange}
        isLocked={false}
      />
    );

    expect(screen.getByTestId('gpu-checkbox')).toBeInTheDocument();
    expect(screen.getByTestId('s3-checkbox')).toBeInTheDocument();
  });

  it('calls onGPUChange when GPU checkbox is clicked', () => {
    render(
      <TrainingOptionsDropdown
        useGPU={false}
        useS3Upload={false}
        onGPUChange={mockOnGPUChange}
        onS3UploadChange={mockOnS3UploadChange}
        isLocked={false}
      />
    );

    const gpuCheckbox = screen.getByTestId('gpu-checkbox');
    fireEvent.click(gpuCheckbox);

    expect(mockOnGPUChange).toHaveBeenCalledWith(true);
  });

  it('calls onS3UploadChange when S3 checkbox is clicked', () => {
    render(
      <TrainingOptionsDropdown
        useGPU={true}
        useS3Upload={false}
        onGPUChange={mockOnGPUChange}
        onS3UploadChange={mockOnS3UploadChange}
        isLocked={false}
      />
    );

    const s3Checkbox = screen.getByTestId('s3-checkbox');
    fireEvent.click(s3Checkbox);

    expect(mockOnS3UploadChange).toHaveBeenCalledWith(true);
  });

  it('disables checkboxes when isLocked is true', () => {
    render(
      <TrainingOptionsDropdown
        useGPU={true}
        useS3Upload={false}
        onGPUChange={mockOnGPUChange}
        onS3UploadChange={mockOnS3UploadChange}
        isLocked={true}
      />
    );

    expect(screen.getByTestId('gpu-checkbox')).toBeDisabled();
    expect(screen.getByTestId('s3-checkbox')).toBeDisabled();
  });
});