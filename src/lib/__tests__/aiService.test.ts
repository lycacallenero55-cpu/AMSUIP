import { AIService } from '../aiService';

// Mock fetch globally
global.fetch = jest.fn();

describe('AIService', () => {
  let aiService: AIService;
  const mockBaseUrl = 'http://localhost:8000';

  beforeEach(() => {
    aiService = new AIService(mockBaseUrl);
    jest.clearAllMocks();
  });

  describe('getTrainedModels', () => {
    it('should fetch trained models without student_id', async () => {
      const mockModels = [
        { id: 1, student_id: 1, model_path: '/models/model1.keras' },
        { id: 2, student_id: 2, model_path: '/models/model2.keras' }
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ models: mockModels })
      });

      const result = await aiService.getTrainedModels();

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/api/training/models`);
      expect(result).toEqual(mockModels);
    });

    it('should fetch trained models with student_id', async () => {
      const mockModels = [{ id: 1, student_id: 1, model_path: '/models/model1.keras' }];
      const studentId = 1;

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ models: mockModels })
      });

      const result = await aiService.getTrainedModels(studentId);

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/api/training/models?student_id=${studentId}`);
      expect(result).toEqual(mockModels);
    });

    it('should handle API errors gracefully', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ detail: 'Not found' })
      });

      const result = await aiService.getTrainedModels();

      expect(result).toEqual([]);
    });
  });

  describe('getGlobalModels', () => {
    it('should fetch global models without limit', async () => {
      const mockModels = [
        { id: 1, model_path: '/models/global1.keras' },
        { id: 2, model_path: '/models/global2.keras' }
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ models: mockModels })
      });

      const result = await aiService.getGlobalModels();

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/api/training/global-models`);
      expect(result).toEqual(mockModels);
    });

    it('should fetch global models with limit', async () => {
      const mockModels = [{ id: 1, model_path: '/models/global1.keras' }];
      const limit = 1;

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ models: mockModels })
      });

      const result = await aiService.getGlobalModels(limit);

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/api/training/global-models?limit=${limit}`);
      expect(result).toEqual(mockModels);
    });
  });

  describe('uploadSignature', () => {
    it('should upload signature successfully', async () => {
      const mockFile = new File(['test'], 'signature.jpg', { type: 'image/jpeg' });
      const mockResponse = {
        record: {
          id: 1,
          student_id: 1,
          label: 'genuine' as const,
          s3_url: 'https://s3.amazonaws.com/bucket/signature.jpg',
          s3_key: 'signatures/1/genuine/signature.jpg'
        }
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await aiService.uploadSignature(1, 'genuine', mockFile);

      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/api/uploads/signature`,
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData)
        })
      );
      expect(result).toEqual(mockResponse.record);
    });

    it('should handle upload errors', async () => {
      const mockFile = new File(['test'], 'signature.jpg', { type: 'image/jpeg' });

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ detail: 'Upload failed' })
      });

      await expect(aiService.uploadSignature(1, 'genuine', mockFile))
        .rejects.toThrow('Upload failed');
    });
  });

  describe('listSignatures', () => {
    it('should list signatures for a student', async () => {
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

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ signatures: mockSignatures })
      });

      const result = await aiService.listSignatures(1);

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/api/uploads/list?student_id=1`);
      expect(result).toEqual(mockSignatures);
    });
  });

  describe('listStudentsWithImages', () => {
    it('should list students with images summary', async () => {
      const mockStudents = [
        { student_id: 1, genuine_count: 5, forged_count: 2 },
        { student_id: 2, genuine_count: 3, forged_count: 1 }
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ items: mockStudents })
      });

      const result = await aiService.listStudentsWithImages(true);

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/api/uploads/students-with-images?summary=true`);
      expect(result).toEqual(mockStudents);
    });

    it('should list students with images without summary', async () => {
      const mockStudents = [
        {
          student_id: 1,
          genuine_count: 5,
          forged_count: 2,
          signatures: [
            { label: 'genuine' as const, s3_url: 'https://s3.amazonaws.com/bucket/sig1.jpg' }
          ]
        }
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ items: mockStudents })
      });

      const result = await aiService.listStudentsWithImages(false);

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/api/uploads/students-with-images`);
      expect(result).toEqual(mockStudents);
    });
  });

  describe('verifySignature', () => {
    it('should verify signature successfully', async () => {
      const mockFile = new File(['test'], 'signature.jpg', { type: 'image/jpeg' });
      const mockResponse = {
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
        decision: 'match' as const,
        message: 'Match found'
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await aiService.verifySignature(mockFile);

      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/api/verification/identify`,
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData)
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should handle verification errors', async () => {
      const mockFile = new File(['test'], 'signature.jpg', { type: 'image/jpeg' });

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ message: 'Verification failed' })
      });

      const result = await aiService.verifySignature(mockFile);

      expect(result).toEqual({
        success: false,
        match: false,
        predicted_student_id: null,
        score: 0,
        decision: 'error',
        message: 'Failed to verify signature',
        error: 'Verification failed'
      });
    });
  });

  describe('startAsyncTraining', () => {
    it('should start async training successfully', async () => {
      const mockFiles = [
        new File(['test1'], 'genuine1.jpg', { type: 'image/jpeg' }),
        new File(['test2'], 'genuine2.jpg', { type: 'image/jpeg' })
      ];
      const mockForgedFiles = [
        new File(['test3'], 'forged1.jpg', { type: 'image/jpeg' })
      ];

      const mockResponse = {
        success: true,
        job_id: 'job-123',
        message: 'Training started',
        stream_url: 'http://localhost:8000/api/progress/stream/job-123',
        training_type: 'hybrid'
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await aiService.startAsyncTraining('STU001', mockFiles, mockForgedFiles, 'hybrid');

      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/api/training/start-async`,
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData)
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should handle training errors', async () => {
      const mockFiles = [new File(['test'], 'genuine1.jpg', { type: 'image/jpeg' })];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve({ detail: 'Training failed' })
      });

      await expect(aiService.startAsyncTraining('STU001', mockFiles, [], 'hybrid'))
        .rejects.toThrow('Training failed');
    });
  });

  describe('startGPUTraining', () => {
    it('should start GPU training successfully', async () => {
      const mockFiles = [new File(['test'], 'genuine1.jpg', { type: 'image/jpeg' })];

      const mockResponse = {
        success: true,
        job_id: 'gpu-job-123',
        message: 'GPU training started',
        stream_url: 'http://localhost:8000/api/progress/stream/gpu-job-123',
        training_type: 'gpu'
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await aiService.startGPUTraining('STU001', mockFiles, [], true);

      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/api/training/start-gpu-training`,
        expect.objectContaining({
          method: 'POST',
          body: expect.any(FormData)
        })
      );
      expect(result).toEqual({
        success: true,
        job_id: 'gpu-job-123',
        message: 'GPU training started',
        stream_url: 'http://localhost:8000/api/progress/stream/gpu-job-123',
        training_type: 'gpu'
      });
    });
  });

  describe('getJobStatus', () => {
    it('should get job status successfully', async () => {
      const mockJob = {
        job_id: 'job-123',
        student_id: 1,
        job_type: 'training',
        status: 'running' as const,
        progress: 50,
        current_stage: 'training',
        estimated_time_remaining: 300,
        start_time: '2023-01-01T10:00:00Z',
        end_time: null,
        result: null,
        error: null,
        created_at: '2023-01-01T10:00:00Z'
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockJob)
      });

      const result = await aiService.getJobStatus('job-123');

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/api/progress/job/job-123`);
      expect(result).toEqual(mockJob);
    });
  });

  describe('subscribeToJobProgress', () => {
    it('should create EventSource for job progress', () => {
      const mockEventSource = {
        onmessage: null,
        onerror: null,
        close: jest.fn()
      };

      // Mock EventSource
      (global as any).EventSource = jest.fn().mockImplementation(() => mockEventSource);

      const onUpdate = jest.fn();
      const onError = jest.fn();

      const eventSource = aiService.subscribeToJobProgress('job-123', onUpdate, onError);

      expect(global.EventSource).toHaveBeenCalledWith(`${mockBaseUrl}/api/progress/stream/job-123`);
      expect(eventSource).toBe(mockEventSource);
    });
  });

  describe('healthCheck', () => {
    it('should check AI service health successfully', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'healthy' })
      });

      const result = await aiService.healthCheck();

      expect(global.fetch).toHaveBeenCalledWith(`${mockBaseUrl}/health`);
      expect(result).toEqual({
        status: 'healthy',
        healthy: true
      });
    });

    it('should handle health check errors', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      const result = await aiService.healthCheck();

      expect(result).toEqual({
        status: 'error',
        healthy: false
      });
    });
  });
});