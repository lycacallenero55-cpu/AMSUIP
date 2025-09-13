import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import SignatureAI from '../SignatureAI';
import { aiService } from '@/lib/aiService';
import { fetchStudents } from '@/lib/supabaseService';

// Mock dependencies
jest.mock('@/lib/aiService');
jest.mock('@/lib/supabaseService');
jest.mock('@/hooks/useAuth', () => ({
  useAuth: () => ({
    user: { id: '1', email: 'test@example.com', role: 'admin' },
    isAuthenticated: true,
    loading: false
  })
}));

// Mock the toast hook
jest.mock('@/components/ui/use-toast', () => ({
  useToast: () => ({
    toast: jest.fn()
  })
}));

// Mock the unsaved changes hook
jest.mock('@/hooks/useUnsavedChanges', () => ({
  useUnsavedChanges: () => ({
    hasUnsavedChanges: false,
    showConfirmDialog: false,
    markAsChanged: jest.fn(),
    markAsSaved: jest.fn(),
    handleClose: jest.fn(),
    confirmClose: jest.fn(),
    cancelClose: jest.fn(),
    handleOpenChange: jest.fn()
  })
}));

const mockAiService = aiService as jest.Mocked<typeof aiService>;
const mockFetchStudents = fetchStudents as jest.MockedFunction<typeof fetchStudents>;

const mockStudents = [
  {
    id: 1,
    student_id: 'STU001',
    firstname: 'John',
    surname: 'Doe',
    program: 'Computer Science',
    year: '1st',
    section: 'A',
    created_at: '2023-01-01T00:00:00Z',
    updated_at: '2023-01-01T00:00:00Z'
  },
  {
    id: 2,
    student_id: 'STU002',
    firstname: 'Jane',
    surname: 'Smith',
    program: 'Computer Science',
    year: '1st',
    section: 'A',
    created_at: '2023-01-01T00:00:00Z',
    updated_at: '2023-01-01T00:00:00Z'
  }
];

const mockStudentsWithImages = [
  {
    student_id: 1,
    genuine_count: 5,
    forged_count: 2
  },
  {
    student_id: 2,
    genuine_count: 3,
    forged_count: 1
  }
];

const mockSignatures = [
  {
    id: 1,
    student_id: 1,
    label: 'genuine' as const,
    s3_url: 'https://s3.amazonaws.com/bucket/signature1.jpg',
    s3_key: 'signatures/1/genuine/signature1.jpg'
  },
  {
    id: 2,
    student_id: 1,
    label: 'forged' as const,
    s3_url: 'https://s3.amazonaws.com/bucket/signature2.jpg',
    s3_key: 'signatures/1/forged/signature2.jpg'
  }
];

const renderSignatureAI = () => {
  return render(
    <BrowserRouter>
      <SignatureAI />
    </BrowserRouter>
  );
};

describe('SignatureAI', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetchStudents.mockResolvedValue(mockStudents);
  });

  it('should render the component with initial state', async () => {
    renderSignatureAI();

    expect(screen.getByText('SIGNATURE AI TRAINING & VERIFICATION')).toBeInTheDocument();
    expect(screen.getByText('Model Training')).toBeInTheDocument();
    expect(screen.getByText('Signature Verification')).toBeInTheDocument();
  });

  it('should load students on mount', async () => {
    renderSignatureAI();

    await waitFor(() => {
      expect(mockFetchStudents).toHaveBeenCalled();
    });
  });

  it('should load students with images when clicking the dropdown menu item', async () => {
    mockAiService.listStudentsWithImages.mockResolvedValue(mockStudentsWithImages);
    mockAiService.listSignatures.mockResolvedValue(mockSignatures);

    renderSignatureAI();

    // Wait for initial load
    await waitFor(() => {
      expect(mockFetchStudents).toHaveBeenCalled();
    });

    // Click the dropdown menu trigger
    const dropdownTrigger = screen.getByRole('button', { name: /more vertical/i });
    fireEvent.click(dropdownTrigger);

    // Click "Load students with images"
    const loadStudentsButton = screen.getByText('Load students with images');
    fireEvent.click(loadStudentsButton);

    await waitFor(() => {
      expect(mockAiService.listStudentsWithImages).toHaveBeenCalledWith(true);
    });

    // Should load individual signatures for each student
    await waitFor(() => {
      expect(mockAiService.listSignatures).toHaveBeenCalledTimes(2);
    });
  });

  it('should handle error when loading students with images', async () => {
    mockAiService.listStudentsWithImages.mockRejectedValue(new Error('API Error'));

    renderSignatureAI();

    // Wait for initial load
    await waitFor(() => {
      expect(mockFetchStudents).toHaveBeenCalled();
    });

    // Click the dropdown menu trigger
    const dropdownTrigger = screen.getByRole('button', { name: /more vertical/i });
    fireEvent.click(dropdownTrigger);

    // Click "Load students with images"
    const loadStudentsButton = screen.getByText('Load students with images');
    fireEvent.click(loadStudentsButton);

    await waitFor(() => {
      expect(mockAiService.listStudentsWithImages).toHaveBeenCalledWith(true);
    });
  });

  it('should show no students message when no students with images found', async () => {
    mockAiService.listStudentsWithImages.mockResolvedValue([]);

    renderSignatureAI();

    // Wait for initial load
    await waitFor(() => {
      expect(mockFetchStudents).toHaveBeenCalled();
    });

    // Click the dropdown menu trigger
    const dropdownTrigger = screen.getByRole('button', { name: /more vertical/i });
    fireEvent.click(dropdownTrigger);

    // Click "Load students with images"
    const loadStudentsButton = screen.getByText('Load students with images');
    fireEvent.click(loadStudentsButton);

    await waitFor(() => {
      expect(mockAiService.listStudentsWithImages).toHaveBeenCalledWith(true);
    });
  });

  it('should handle file upload for verification', async () => {
    const mockFile = new File(['test'], 'signature.jpg', { type: 'image/jpeg' });
    mockAiService.verifySignature.mockResolvedValue({
      success: true,
      match: true,
      predicted_student_id: 1,
      predicted_student: {
        id: 1,
        student_id: 'STU001',
        firstname: 'John',
        surname: 'Doe'
      },
      score: 0.95,
      decision: 'match',
      message: 'Match found'
    });

    renderSignatureAI();

    // Find the file input
    const fileInput = screen.getByLabelText(/signature preview/i).parentElement?.querySelector('input[type="file"]');
    expect(fileInput).toBeInTheDocument();

    // Upload a file
    if (fileInput) {
      fireEvent.change(fileInput, { target: { files: [mockFile] } });
    }

    // Click verify button
    const verifyButton = screen.getByText('Verify Signature');
    fireEvent.click(verifyButton);

    await waitFor(() => {
      expect(mockAiService.verifySignature).toHaveBeenCalledWith(mockFile);
    });
  });

  it('should handle camera capture for verification', async () => {
    // Mock getUserMedia
    const mockStream = {
      getTracks: () => [{ stop: jest.fn() }]
    };
    (navigator.mediaDevices as any) = {
      getUserMedia: jest.fn().mockResolvedValue(mockStream)
    };

    // Mock canvas
    const mockCanvas = {
      width: 640,
      height: 480,
      getContext: jest.fn().mockReturnValue({
        drawImage: jest.fn()
      }),
      toBlob: jest.fn().mockImplementation((callback) => {
        callback(new Blob(['test'], { type: 'image/png' }));
      })
    };

    renderSignatureAI();

    // Click camera button
    const cameraButton = screen.getByText('Camera');
    fireEvent.click(cameraButton);

    // Mock video element
    const videoElement = screen.getByRole('video');
    Object.defineProperty(videoElement, 'videoWidth', { value: 640 });
    Object.defineProperty(videoElement, 'videoHeight', { value: 480 });
    Object.defineProperty(videoElement, 'srcObject', { value: mockStream });

    // Mock canvas ref
    const canvasElement = document.createElement('canvas');
    Object.assign(canvasElement, mockCanvas);
    document.body.appendChild(canvasElement);

    // Click capture button
    const captureButton = screen.getByText('Capture');
    fireEvent.click(captureButton);

    await waitFor(() => {
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({ video: true });
    });
  });

  it('should start training when train button is clicked', async () => {
    mockAiService.startGPUTraining.mockResolvedValue({
      success: true,
      job_id: 'job-123',
      message: 'Training started',
      stream_url: 'http://localhost:8000/api/progress/stream/job-123',
      training_type: 'gpu'
    });

    // Mock EventSource
    const mockEventSource = {
      onmessage: null,
      onerror: null,
      close: jest.fn()
    };
    (global as any).EventSource = jest.fn().mockImplementation(() => mockEventSource);

    renderSignatureAI();

    // Wait for initial load
    await waitFor(() => {
      expect(mockFetchStudents).toHaveBeenCalled();
    });

    // Load students with images first
    mockAiService.listStudentsWithImages.mockResolvedValue(mockStudentsWithImages);
    mockAiService.listSignatures.mockResolvedValue(mockSignatures);

    // Click the dropdown menu trigger
    const dropdownTrigger = screen.getByRole('button', { name: /more vertical/i });
    fireEvent.click(dropdownTrigger);

    // Click "Load students with images"
    const loadStudentsButton = screen.getByText('Load students with images');
    fireEvent.click(loadStudentsButton);

    await waitFor(() => {
      expect(mockAiService.listStudentsWithImages).toHaveBeenCalledWith(true);
    });

    // Wait for students to be loaded
    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });

    // Click train button
    const trainButton = screen.getByText('Train Model');
    fireEvent.click(trainButton);

    await waitFor(() => {
      expect(mockAiService.startGPUTraining).toHaveBeenCalled();
    });
  });

  it('should show training progress when training is active', async () => {
    mockAiService.startGPUTraining.mockResolvedValue({
      success: true,
      job_id: 'job-123',
      message: 'Training started',
      stream_url: 'http://localhost:8000/api/progress/stream/job-123',
      training_type: 'gpu'
    });

    // Mock EventSource
    const mockEventSource = {
      onmessage: null,
      onerror: null,
      close: jest.fn()
    };
    (global as any).EventSource = jest.fn().mockImplementation(() => mockEventSource);

    renderSignatureAI();

    // Load students with images first
    mockAiService.listStudentsWithImages.mockResolvedValue(mockStudentsWithImages);
    mockAiService.listSignatures.mockResolvedValue(mockSignatures);

    // Click the dropdown menu trigger
    const dropdownTrigger = screen.getByRole('button', { name: /more vertical/i });
    fireEvent.click(dropdownTrigger);

    // Click "Load students with images"
    const loadStudentsButton = screen.getByText('Load students with images');
    fireEvent.click(loadStudentsButton);

    await waitFor(() => {
      expect(mockAiService.listStudentsWithImages).toHaveBeenCalledWith(true);
    });

    // Wait for students to be loaded
    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });

    // Click train button
    const trainButton = screen.getByText('Train Model');
    fireEvent.click(trainButton);

    await waitFor(() => {
      expect(screen.getByText(/Training.../)).toBeInTheDocument();
    });
  });

  it('should handle training errors', async () => {
    mockAiService.startGPUTraining.mockRejectedValue(new Error('Training failed'));

    renderSignatureAI();

    // Load students with images first
    mockAiService.listStudentsWithImages.mockResolvedValue(mockStudentsWithImages);
    mockAiService.listSignatures.mockResolvedValue(mockSignatures);

    // Click the dropdown menu trigger
    const dropdownTrigger = screen.getByRole('button', { name: /more vertical/i });
    fireEvent.click(dropdownTrigger);

    // Click "Load students with images"
    const loadStudentsButton = screen.getByText('Load students with images');
    fireEvent.click(loadStudentsButton);

    await waitFor(() => {
      expect(mockAiService.listStudentsWithImages).toHaveBeenCalledWith(true);
    });

    // Wait for students to be loaded
    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });

    // Click train button
    const trainButton = screen.getByText('Train Model');
    fireEvent.click(trainButton);

    await waitFor(() => {
      expect(mockAiService.startGPUTraining).toHaveBeenCalled();
    });
  });

  it('should disable train button when no students are selected', () => {
    renderSignatureAI();

    const trainButton = screen.getByText('Train Model');
    expect(trainButton).toBeDisabled();
  });

  it('should enable train button when students with images are loaded', async () => {
    mockAiService.listStudentsWithImages.mockResolvedValue(mockStudentsWithImages);
    mockAiService.listSignatures.mockResolvedValue(mockSignatures);

    renderSignatureAI();

    // Wait for initial load
    await waitFor(() => {
      expect(mockFetchStudents).toHaveBeenCalled();
    });

    // Click the dropdown menu trigger
    const dropdownTrigger = screen.getByRole('button', { name: /more vertical/i });
    fireEvent.click(dropdownTrigger);

    // Click "Load students with images"
    const loadStudentsButton = screen.getByText('Load students with images');
    fireEvent.click(loadStudentsButton);

    await waitFor(() => {
      expect(mockAiService.listStudentsWithImages).toHaveBeenCalledWith(true);
    });

    // Wait for students to be loaded
    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });

    const trainButton = screen.getByText('Train Model');
    expect(trainButton).not.toBeDisabled();
  });
});