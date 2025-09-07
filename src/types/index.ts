// Signature types
export interface Signature {
  id: number;
  student_id: number;
  storage_path: string;
  file_name: string;
  file_size: number;
  file_type: string;
  width?: number;
  height?: number;
  features?: Record<string, unknown>;
  quality_score?: number;
  device_info?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface DeviceInfo {
  userAgent: string;
  screenResolution: string;
  timezone: string;
  [key: string]: unknown;
}

export interface FileInfo {
  size: number;
  type: string;
  dimensions?: { width: number; height: number };
  [key: string]: unknown;
}

export interface SignatureUploadOptions {
  deviceInfo?: DeviceInfo;
  fileInfo?: FileInfo;
  features?: Record<string, unknown>;
  quality_score?: number;
  width?: number;
  height?: number;
  onProgress?: (progress: number) => void;
}

export interface SignatureMatchResult {
  signature: Signature;
  score: number;
  isMatch: boolean;
}

// Base types
export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'staff' | 'ssg_officer';
  avatar_url?: string;
  created_at?: string;
  updated_at?: string;
}

export interface Student {
  id: number;
  student_id: string;
  firstname: string;
  surname: string;
  program: string;
  year: string;
  section: string;
  signature_url?: string;
  created_at: string;
  updated_at: string;
  full_name?: string; // Computed field
}

export interface Session {
  id: number;
  title: string;
  type: 'class' | 'event' | 'other';
  time?: string; // Kept for backward compatibility
  time_in?: string;
  time_out?: string;
  created_by_user_id?: string; // UUID of the user who created the session
  students?: number;
  program: string;
  year: string;
  section: string;
  description?: string;
  capacity: string | number;
  date: string;
  created_at: string;
  updated_at: string;
}

export interface Subject {
  id: number;
  code: string;
  name: string;
  description?: string;
  units: number;
  program: string;
  year_level: string;
  semester: '1st' | '2nd' | 'Summer';
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface AttendanceRecord {
  id: number;
  session_id: number;
  student_id: number;
  status: 'present' | 'absent' | 'late' | 'excused';
  time_in?: string;
  time_out?: string;
  created_at: string;
  updated_at: string;
}

// Response types
export interface SessionWithStudents {
  session: Pick<Session, 'id' | 'title' | 'date' | 'time_in' | 'time_out' | 'program' | 'year' | 'section' | 'description' | 'type'>;
  students: Array<Student & {
    status: AttendanceRecord['status'] | null;
    time_in: string | null;
    time_out: string | null;
  }>;
  count: number;
}

// Auth types
export interface AuthResponse {
  user: User | null;
  session: {
    access_token: string;
    refresh_token: string;
    expires_in: number;
    token_type: string;
    user: User;
  } | null;
  error: Error | null;
}

export interface AuthContextType {
  user: User | null;
  loading: boolean;
  error: Error | null;
  signIn: (email: string, password: string) => Promise<AuthResponse>;
  signUp: (email: string, password: string, userData: Partial<User>) => Promise<AuthResponse>;
  signOut: () => Promise<{ error: Error | null }>;
  resetPassword: (email: string) => Promise<{ error: Error | null }>;
  updatePassword: (newPassword: string) => Promise<{ error: Error | null }>;
  updateProfile: (updates: Partial<User>) => Promise<{ user: User | null; error: Error | null }>;
  isAuthenticated: () => boolean;
  hasRole: (role: User['role']) => boolean;
}

// API response types
export interface ApiResponse<T> {
  data: T | null;
  error: Error | null;
  status: number;
  statusText: string;
}

// Form types
export interface LoginFormData {
  email: string;
  password: string;
}

export interface SignUpFormData extends LoginFormData {
  name: string;
  confirmPassword: string;
}

// Table types
export interface Column<T> {
  id: keyof T | 'actions';
  label: string;
  minWidth?: number;
  align?: 'right' | 'left' | 'center';
  format?: <K extends keyof T>(value: T[K], row?: T) => string | JSX.Element;
  sortable?: boolean;
}

// Training types
export interface TrainingFile {
  file: File;
  preview: string;
  id?: number; // signature record id (DB)
  s3Key?: string;
  label?: 'genuine' | 'forged';
}

export interface StudentTrainingCard {
  id: string;
  student: Student | null;
  genuineFiles: TrainingFile[];
  forgedFiles: TrainingFile[];
  isExpanded: boolean;
}

// Utility types
export type WithRequired<T, K extends keyof T> = T & { [P in K]-?: T[P] };