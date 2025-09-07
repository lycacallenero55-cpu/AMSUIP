// AI Service Configuration and Client

const AI_BASE_URL = import.meta.env.VITE_AI_BASE_URL || 'http://localhost:8000';

export interface AITrainingResponse {
  success: boolean;
  message: string;
  profile?: {
    student_id: number;
    status: 'untrained' | 'training' | 'ready' | 'error';
    embedding_centroid: number[] | null;
    num_samples: number;
    threshold: number;
    last_trained_at: string | null;
    error_message: string | null;
  };
  error?: string;
}

export interface AIVerificationResponse {
  success: boolean;
  match: boolean;
  predicted_student_id: number | null;
  predicted_student?: {
    id: number;
    student_id: string;
    firstname: string;
    surname: string;
  };
  score: number;
}

export interface AsyncTrainingResponse {
  success: boolean;
  job_id: string;
  message: string;
  stream_url: string;
}

export interface TrainingJob {
  job_id: string;
  student_id: number;
  job_type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_stage: string;
  estimated_time_remaining: number | null;
  start_time: string | null;
  end_time: string | null;
  result: any;
  error: string | null;
  created_at: string;
  training_metrics?: {
    current_epoch: number;
    total_epochs: number;
    accuracy: number;
    val_accuracy: number;
    loss: number;
    val_loss: number;
    precision: number;
    recall: number;
    auc: number;
    val_auc: number;
    learning_rate: number;
    batch_progress: string;
    epoch_progress: string;
    current_batch?: number;
  };
}

export interface AIVerificationResponse {
  success: boolean;
  match: boolean;
  predicted_student_id: number | null;
  predicted_student?: {
    id: number;
    student_id: string;
    firstname: string;
    surname: string;
  };
  score: number;
  decision: 'match' | 'no_match' | 'error';
  message: string;
  error?: string;
}

export class AIService {
  private baseUrl: string;

  constructor(baseUrl: string = AI_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Get trained models (optionally by student_id)
   */
  async getTrainedModels(studentId?: number): Promise<any[]> {
    try {
      const url = studentId
        ? `${this.baseUrl}/api/training/models?student_id=${studentId}`
        : `${this.baseUrl}/api/training/models`;
      const response = await fetch(url);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to fetch models');
      }
      return data.models || [];
    } catch (error) {
      console.error('AI getTrainedModels error:', error);
      return [];
    }
  }

  /**
   * Train AI model for a specific student
   */
  async trainStudent(studentId: number): Promise<AITrainingResponse> {
    try {
      // Keep method for backward compatibility; recommend using trainStudentWithFiles
      const response = await fetch(`${this.baseUrl}/api/training/start`, {
        method: 'POST'
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Training request failed');
      }

      return data;
    } catch (error) {
      console.error('AI training error:', error);
      return {
        success: false,
        message: 'Failed to start training',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Train with actual files against FastAPI endpoint
   */
  async trainStudentWithFiles(
    studentSchoolId: string,
    genuineFiles: File[],
    forgedFiles: File[]
  ): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('student_id', String(studentSchoolId));
      for (const f of genuineFiles) formData.append('genuine_files', f);
      for (const f of forgedFiles) formData.append('forged_files', f);

      const response = await fetch(`${this.baseUrl}/api/training/start`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Training request failed');
      }
      // Normalize to include success flag for UI consistency
      return { success: true, ...data };
    } catch (error) {
      console.error('AI training (files) error:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Verify signature image
   */
  async verifySignature(
    imageFile: File,
    sessionId?: number
  ): Promise<AIVerificationResponse> {
    try {
      const formData = new FormData();
      // Backend expects 'test_file' for verification/identify endpoints
      formData.append('test_file', imageFile);
      
      if (sessionId) {
        formData.append('session_id', sessionId.toString());
      }

      const response = await fetch(`${this.baseUrl}/api/verification/identify`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Verification request failed');
      }

      return data;
    } catch (error) {
      console.error('AI verification error:', error);
      return {
        success: false,
        match: false,
        predicted_student_id: null,
        score: 0,
        decision: 'error',
        message: 'Failed to verify signature',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Get a browser-displayable preview PNG for any uploaded image (e.g., TIFF)
   */
  async getPreviewURL(file: File): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);
    const resp = await fetch(`${this.baseUrl}/api/utils/preview`, { method: 'POST', body: formData });
    if (!resp.ok) {
      throw new Error('Failed to generate preview');
    }
    const blob = await resp.blob();
    return URL.createObjectURL(blob);
  }

  /**
   * Verify using model and references (FastAPI shape)
   */
  async verifyWithModel(
    modelId: string,
    referenceFiles: File[],
    testFile: File
  ): Promise<AIVerificationResponse> {
    try {
      const formData = new FormData();
      formData.append('model_id', modelId);
      for (const f of referenceFiles) formData.append('reference_files', f);
      formData.append('test_file', testFile);

      const response = await fetch(`${this.baseUrl}/api/verification/verify`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Verification failed');
      }
      return data;
    } catch (error) {
      console.error('AI verification (model) error:', error);
      return {
        success: false,
        match: false,
        predicted_student_id: null,
        score: 0,
        decision: 'error',
        message: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Verify signature from data URL (base64 encoded image)
   */
  async verifySignatureFromDataURL(
    dataURL: string,
    sessionId?: number
  ): Promise<AIVerificationResponse> {
    try {
      // Convert data URL to blob
      const response = await fetch(dataURL);
      const blob = await response.blob();
      
      // Create file from blob
      const file = new File([blob], 'signature.png', { type: 'image/png' });
      
      return this.verifySignature(file, sessionId);
    } catch (error) {
      console.error('Error converting data URL to file:', error);
      return {
        success: false,
        match: false,
        predicted_student_id: null,
        score: 0,
        decision: 'error',
        message: 'Failed to process signature image',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Start async training job
   */
  async startAsyncTraining(
    studentId: string,
    genuineFiles: File[],
    forgedFiles: File[]
  ): Promise<AsyncTrainingResponse> {
    try {
      const formData = new FormData();
      formData.append('student_id', studentId);
      for (const file of genuineFiles) {
        formData.append('genuine_files', file);
      }
      for (const file of forgedFiles) {
        formData.append('forged_files', file);
      }

      const response = await fetch(`${this.baseUrl}/api/training/start-async`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        console.error('Training API error:', {
          status: response.status,
          statusText: response.statusText,
          data: data
        });
        throw new Error(data.detail || data.message || `Training failed with status ${response.status}`);
      }
      return data;
    } catch (error) {
      console.error('AI async training error:', error);
      throw error;
    }
  }

  /**
   * Get training job status
   */
  async getJobStatus(jobId: string): Promise<TrainingJob> {
    try {
      const response = await fetch(`${this.baseUrl}/api/progress/job/${jobId}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to get job status');
      }
      return data;
    } catch (error) {
      console.error('Get job status error:', error);
      throw error;
    }
  }

  /**
   * Subscribe to training job progress via Server-Sent Events
   */
  subscribeToJobProgress(
    jobId: string,
    onUpdate: (job: TrainingJob) => void,
    onError?: (error: Error) => void
  ): EventSource {
    const eventSource = new EventSource(`${this.baseUrl}/api/progress/stream/${jobId}`);
    
    eventSource.onmessage = (event) => {
      try {
        const job: TrainingJob = JSON.parse(event.data);
        onUpdate(job);
      } catch (error) {
        console.error('Error parsing SSE data:', error);
        if (onError) {
          onError(error instanceof Error ? error : new Error('Parse error'));
        }
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      if (onError) {
        onError(new Error('Connection error'));
      }
    };
    
    return eventSource;
  }

  /**
   * Check AI service health
   */
  async healthCheck(): Promise<{ status: string; healthy: boolean }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      
      return {
        status: data.status || 'unknown',
        healthy: response.ok && data.status === 'healthy',
      };
    } catch (error) {
      console.error('AI service health check failed:', error);
      return {
        status: 'error',
        healthy: false,
      };
    }
  }
}

// Export a singleton instance
export const aiService = new AIService();

// Export configuration
export const AI_CONFIG = {
  BASE_URL: AI_BASE_URL,
  ENDPOINTS: {
    TRAIN: '/train',
    VERIFY: '/verify',
    HEALTH: '/health',
  },
};