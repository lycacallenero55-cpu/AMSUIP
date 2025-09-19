import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { format, addDays, startOfWeek, endOfWeek } from 'date-fns';
import { 
  BookOpen, 
  Calendar,
  Clock, 
  Loader2, 
  Plus, 
  Search,
  Star, 
  Users, 
  ChevronLeft, 
  ChevronRight, 
  MapPin,
  SquarePen,
  Trash2,
  CalendarClock
} from 'lucide-react';

// UI Components
import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogFooter, 
  DialogDescription 
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from "@/components/ui/table";
import { useToast } from "@/components/ui/use-toast";

// Custom Components and Services
import AttendanceForm from "@/components/AttendanceForm";
import { 
  fetchSessions, 
  createSession, 
  updateSession, 
  deleteSession 
} from "@/lib/supabaseService";
import { supabase } from "@/lib/supabase";

// Types
import type { SessionData } from "@/components/AttendanceForm";
import type { Session as SessionType } from "@/types";

// Define types for the attendance data from Supabase
type Student = {
  program: string;
  year: string;
  section: string;
};

type AttendanceRecord = {
  student_id: string;
  students: Student[];
};

// Type for the raw response from Supabase
type SupabaseAttendanceResponse = Array<{
  student_id: string;
  students: Student;
}>;

// Extend the base Session type with our local requirements
interface Session extends Omit<SessionType, 'time_in' | 'time_out' | 'capacity'> {
  time: string; // Combined time display
  time_in: string;
  time_out: string;
  location: string;
  instructor: string;
  students: number;
  present?: number;
  absent?: number;
  program: string;
  year: string;
  section: string;
  capacity: string; // Changed to only allow string to match the database schema
  date: string;
}

// Helper function to format date as YYYY-MM-DD in local timezone
// Format time to 12-hour format with AM/PM
const formatTime = (timeString: string) => {
  if (!timeString) return '--:--';
  
  // Handle both 'HH:mm' and 'HH:mm:ss' formats
  const [hours, minutes] = timeString.split(':');
  const hour = parseInt(hours, 10);
  const mins = minutes || '00';
  
  const period = hour >= 12 ? 'PM' : 'AM';
  const displayHour = hour % 12 || 12; // Convert 0 to 12 for 12 AM
  
  return `${displayHour}:${mins} ${period}`;
};

const formatDateString = (date: Date | string): string => {
  try {
    // If input is already a string in YYYY-MM-DD format, return as is
    if (typeof date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(date)) {
      return date;
    }
    
    // Create a new date object to avoid mutating the original
    const d = new Date(date);
    
    // Check if the date is valid
    if (isNaN(d.getTime())) {
      console.error('Invalid date provided to formatDateString:', date);
      // Return today's date as fallback
      const today = new Date();
      return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    }
    
    // Use UTC methods to avoid timezone issues
    const year = d.getUTCFullYear();
    const month = String(d.getUTCMonth() + 1).padStart(2, '0');
    const day = String(d.getUTCDate()).padStart(2, '0');
    
    const formattedDate = `${year}-${month}-${day}`;
    console.log('Formatted date:', { input: date, output: formattedDate });
    return formattedDate;
  } catch (error) {
    console.error('Error in formatDateString:', error);
    // Return today's date as fallback
    const today = new Date();
    return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  }
};

// Helper function to get week dates for a given date
const getWeekDates = (date: Date): Date[] => {
  try {
    // Create a copy of the input date to avoid modifying it
    const inputDate = new Date(date);
    
    // Get the day of the week (0 = Sunday, 1 = Monday, etc.)
    const dayOfWeek = inputDate.getUTCDay();
    
    // Calculate the date of the previous Sunday
    const sunday = new Date(inputDate);
    sunday.setUTCDate(inputDate.getUTCDate() - dayOfWeek);
    
    // Create an array to hold all 7 days of the week
    const weekDates: Date[] = [];
    
    // Generate all 7 days of the week starting from Sunday
    for (let i = 0; i < 7; i++) {
      // Create a new date for each day of the week using UTC methods
      const day = new Date(Date.UTC(
        sunday.getUTCFullYear(),
        sunday.getUTCMonth(),
        sunday.getUTCDate() + i,
        0, 0, 0, 0  // Set time to midnight UTC
      ));
      
      weekDates.push(day);
    }
    
    // Debug log the week dates
    console.log('Generated week dates:');
    weekDates.forEach((d, i) => {
      console.log(`Day ${i}:`, d.toISOString().split('T')[0]);
    });
    
    return weekDates;
  } catch (error) {
    console.error('Error in getWeekDates:', error);
    // Return current week as fallback
    const today = new Date();
    const dayOfWeek = today.getUTCDay();
    const sunday = new Date(today);
    sunday.setUTCDate(today.getUTCDate() - dayOfWeek);
    
    return Array.from({ length: 7 }, (_, i) => {
      const d = new Date(sunday);
      d.setUTCDate(sunday.getUTCDate() + i);
      return d;
    });
  }
};

// Extend SessionData to include id for editing
type SessionDataWithId = SessionData & { id: number };

const Schedule = () => {
  // State declarations at the top level
  const { toast } = useToast();
  const navigate = useNavigate();
  
  // Get today's date at midnight in local time
  const today = useMemo(() => {
    // Create a new date object with the current date in local timezone
    const now = new Date();
    // Create a new date with just the date components (no time)
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    return today;
  }, []);
  
  // Component state
  const [currentDate, setCurrentDate] = useState<Date>(today);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingSession, setEditingSession] = useState<SessionDataWithId | null>(null);
  const [sessionToDelete, setSessionToDelete] = useState<number | null>(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [programs, setPrograms] = useState<string[]>([]);
  const [isLoadingPrograms, setIsLoadingPrograms] = useState<boolean>(true);
  
  // Search and filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [programFilter, setProgramFilter] = useState('all');
  const [dateFilter, setDateFilter] = useState('all');
  
  // Pagination state
  const [pagination, setPagination] = useState({
    currentPage: 1,
    pageSize: 10,
    totalItems: 0
  });
  const [displayPageSize, setDisplayPageSize] = useState(10);
  const [totalSessionsCount, setTotalSessionsCount] = useState(0);
  
  // Derived state - week dates for the current view
  const weekDates = useMemo(() => getWeekDates(currentDate), [currentDate]);
  
  // Handle page size change
  const handlePageSizeChange = (value: string) => {
    const newPageSize = value === 'all' ? totalSessionsCount : parseInt(value);
    setPagination(prev => ({
      ...prev,
      pageSize: newPageSize,
      currentPage: 1
    }));
    setDisplayPageSize(newPageSize);
  };
  
  // Reset form when editing session changes
  useEffect(() => {
    if (editingSession) {
      setIsModalOpen(true);
    }
  }, [editingSession, setIsModalOpen]);
  
  // Update displayPageSize when totalSessionsCount changes and "ALL" is selected
  useEffect(() => {
    if (displayPageSize === totalSessionsCount && totalSessionsCount > 0) {
      setDisplayPageSize(totalSessionsCount);
    }
  }, [totalSessionsCount, displayPageSize]);

  // Format date for display using the browser's local timezone
  const formattedCurrentDate = useMemo(() => {
    // Use the current date's local time components
    const options: Intl.DateTimeFormatOptions = { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    };
    return currentDate.toLocaleDateString('en-US', options);
  }, [currentDate]);
  
  // Memoize the loadSessions function to prevent unnecessary re-renders
  const loadSessions = useCallback(async (dates?: Date[], minDate?: string) => {
    console.log('loadSessions called with:', { dates, minDate });
    setIsLoading(true);
    setError(null);
    
    try {
      // 1. First, fetch all sessions with the date range filter
      let query = supabase
        .from('sessions')
        .select('*')
        .order('date', { ascending: true })
        .order('time_in', { ascending: true });
      
      // Apply date range filter based on the parameters
      if (minDate) {
        // If minDate is provided, only get sessions on or after this date
        query = query.gte('date', minDate);
      } else if (dates && dates.length > 0) {
        // If specific dates are provided, use those as the range
        const startDate = formatDateString(dates[0]);
        const endDate = formatDateString(dates[dates.length - 1]);
        query = query.gte('date', startDate).lte('date', endDate);
      }
      
      const { data: sessions, error: sessionsError } = await query;
      
      if (sessionsError) throw sessionsError;
      if (!sessions || sessions.length === 0) {
        setSessions([]);
        return;
      }
      
      // 2. Get unique program, year, section combinations to minimize queries
      const uniqueCombinations = new Set<string>();
      
      // Process each session to create unique combinations
      sessions.forEach(session => {
        const program = session.program || 'All Programs';
        const year = session.year || 'All Year Levels';
        const section = session.section || 'All Sections';
        uniqueCombinations.add(`${program}::${year}::${section}`);
      });
      
      // 3. Fetch student counts for each unique combination in a single batch
      const studentCountPromises = Array.from(uniqueCombinations).map(async (combo) => {
        const [program, year, section] = combo.split('::');
        
        let query = supabase
          .from('students')
          .select('*', { count: 'exact', head: true });
          
        if (program && !program.toLowerCase().includes('all')) {
          query = query.eq('program', program);
        }
        if (year && !year.toLowerCase().includes('all')) {
          // Convert year format for consistent matching
          let yearValue = year;
          // Handle both 'All Years' and 'All Year Levels' variants
          if (yearValue === 'All Years' || yearValue === 'All Year Levels') {
            // Skip the filter for 'All' selections
          } else {
            // Convert '1st Year' to '1st' for database query
            if (yearValue.endsWith(' Year')) {
              yearValue = yearValue.replace(' Year', '');
            }
            query = query.eq('year', yearValue);
          }
        }
        if (section && !section.toLowerCase().includes('all')) {
          query = query.eq('section', section);
        }
        
        const { count } = await query;
        return { program, year, section, count: count || 0 };
      });
      
      // 4. Wait for all student count queries to complete
      const studentCounts = await Promise.all(studentCountPromises);
      
      // 5. Create a map for quick lookup of student counts
      const studentCountMap = new Map();
      studentCounts.forEach(({ program, year, section, count }) => {
        studentCountMap.set(`${program}::${year}::${section}`, count);
      });
      
      // 6. Fetch attendance counts per session and format
      const attendanceCountsPromises = sessions.map(async (s) => {
        const { data: att, error: attErr } = await supabase
          .from('attendance')
          .select('status', { count: 'exact' })
          .eq('session_id', s.id);
        if (attErr) return { sessionId: s.id, present: 0, absent: 0 };
        const present = (att || []).filter((a: any) => a.status === 'present').length;
        const absent = (att || []).filter((a: any) => a.status === 'absent').length;
        return { sessionId: s.id, present, absent };
      });
      const attendanceCounts = await Promise.all(attendanceCountsPromises);
      const attendanceMap = new Map(attendanceCounts.map(a => [a.sessionId, a]));

      const formattedSessions = sessions.map(session => {
        const program = session.program || 'All Programs';
        const year = session.year || 'All Year Levels';
        const section = session.section || 'All Sections';
        const sessionKey = `${program}::${year}::${section}`;
        const studentCount = studentCountMap.get(sessionKey) || 0;
        const att = attendanceMap.get(session.id) || { present: 0, absent: 0 };
        
        return {
          id: session.id,
          title: session.title || 'Untitled Session',
          type: (session.type as 'class' | 'event' | 'other') || 'class',
          time_in: session.time_in || '',
          time_out: session.time_out || '',
          time: session.time_in && session.time_out 
            ? `${formatTime(session.time_in)} - ${formatTime(session.time_out)}` 
            : '',
          students: studentCount,
          present: att.present,
          absent: att.absent,
          program: session.program || 'General',
          year: session.year || 'All Year Levels',
          section: session.section || 'All Sections',
          description: session.description || '',
          capacity: session.capacity ? String(session.capacity) : 'Unlimited',
          date: session.date,
          created_at: session.created_at || new Date().toISOString(),
          updated_at: session.updated_at || new Date().toISOString()
        } as Session;
      });
      
      setSessions(formattedSessions);
      setTotalSessionsCount(formattedSessions.length);
    } catch (err) {
      console.error('Failed to load sessions:', err);
      setError('Failed to load sessions. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch programs from the database with pagination
  const fetchPrograms = useCallback(async () => {
    try {
      setIsLoadingPrograms(true);
      
      let allStudents: { program: string | null }[] = [];
      let page = 0;
      const pageSize = 1000;
      let hasMore = true;
      
      // Fetch all students with pagination
      while (hasMore) {
        const { data, error, count } = await supabase
          .from('students')
          .select('program', { count: 'exact' })
          .not('program', 'is', null)
          .range(page * pageSize, (page + 1) * pageSize - 1);
          
        if (error) throw error;
        
        if (data && data.length > 0) {
          allStudents = [...allStudents, ...data];
          page++;
          
          // If we got fewer items than requested, we've reached the end
          if (data.length < pageSize) hasMore = false;
        } else {
          hasMore = false;
        }
      }
      
      console.log(`Fetched ${allStudents.length} students with programs`);
      
      // Process programs
      const programSet = new Set<string>();
      allStudents.forEach(student => {
        if (student.program) {
          const program = student.program.toString().trim();
          if (program) programSet.add(program);
        }
      });
      
      const uniquePrograms = Array.from(programSet).sort((a, b) => 
        a.localeCompare(b, 'en', { sensitivity: 'base' })
      );
      
      console.log('Unique programs found:', uniquePrograms);
      setPrograms(uniquePrograms);
    } catch (err) {
      console.error('Error fetching programs:', err);
      setPrograms([]); // Set empty array on error
    } finally {
      setIsLoadingPrograms(false);
    }
  }, []);

  // Fetch programs when component mounts
  useEffect(() => {
    fetchPrograms();
  }, [fetchPrograms]);

  // Memoize loadSessions to prevent unnecessary re-renders
  const loadSessionsMemoized = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Fetch sessions from Supabase
      const { data: sessions, error } = await supabase
        .from('sessions')
        .select('*')
        .order('date', { ascending: true });
        
      if (error) throw error;
      
      // Get unique program, year, section combinations to minimize queries
      const uniqueCombinations = new Set(
        sessions
          .filter(s => s.program && s.year && s.section)
          .map(s => `${s.program}::${s.year}::${s.section}`)
      );
      
      // Fetch student counts for each unique combination in a single batch
      const studentCountPromises = Array.from(uniqueCombinations).map(async (combo) => {
        const [program, year, section] = combo.split('::');
        
        let query = supabase
          .from('students')
          .select('*', { count: 'exact', head: true });
          
        if (program && !program.toLowerCase().includes('all')) {
          query = query.eq('program', program);
        }
        if (year && !year.toLowerCase().includes('all')) {
          // Convert year format for consistent matching
          let yearValue = year;
          if (yearValue === 'All Years' || yearValue === 'All Year Levels') {
            // Skip the filter for 'All' selections
          } else {
            // Convert '1st Year' to '1st' for database query
            if (yearValue.endsWith(' Year')) {
              yearValue = yearValue.replace(' Year', '');
            }
            query = query.eq('year', yearValue);
          }
        }
        if (section && !section.toLowerCase().includes('all')) {
          query = query.eq('section', section);
        }
        
        const { count } = await query;
        return { program, year, section, count: count || 0 };
      });
      
      // Wait for all student count queries to complete
      const studentCounts = await Promise.all(studentCountPromises);
      
      // Create a map for quick lookup of student counts
      const studentCountMap = new Map();
      studentCounts.forEach(({ program, year, section, count }) => {
        studentCountMap.set(`${program}::${year}::${section}`, count);
      });
      
      // Format sessions with student counts
      const formattedSessions = sessions.map(session => {
        const sessionKey = `${session.program || 'all'}::${session.year || 'all'}::${session.section || 'all'}`;
        const studentCount = studentCountMap.get(sessionKey) || 0;
        
        return {
          id: session.id,
          title: session.title || 'Untitled Session',
          type: (session.type as 'class' | 'event' | 'other') || 'class',
          time_in: session.time_in || '',
          time_out: session.time_out || '',
          time: session.time_in && session.time_out 
            ? `${formatTime(session.time_in)} - ${formatTime(session.time_out)}` 
            : '',
          students: studentCount,
          program: session.program || 'General',
          year: session.year || 'All Year Levels',
          section: session.section || 'All Sections',
          description: session.description || '',
          capacity: session.capacity ? String(session.capacity) : 'Unlimited',
          date: session.date,
          created_at: session.created_at || new Date().toISOString(),
          updated_at: session.updated_at || new Date().toISOString()
        } as Session;
      });
      
      setSessions(formattedSessions);
      setTotalSessionsCount(formattedSessions.length);
    } catch (err) {
      console.error('Failed to load sessions:', err);
      setError('Failed to load sessions. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  }, []); // Add formatTime as a dependency since it's used in the callback

  // Load sessions when component mounts and when weekDates or dateFilter changes
  useEffect(() => {
    let isMounted = true;
    
    const loadData = async () => {
      try {
        // Always load all sessions by default
        console.log('Loading all sessions');
        await loadSessionsMemoized(); // Use the memoized version
      } catch (err) {
        if (isMounted) {
          console.error('Error in loadSessions:', err);
          setError('Failed to load sessions. Please try again.');
          setIsLoading(false);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };
    
    loadData();
    
    return () => {
      isMounted = false;
    };
  }, [weekDates, dateFilter, loadSessionsMemoized]); // Removed isModalOpen from dependencies

  // Memoize today's date and pre-compute date ranges
  const dateRanges = useMemo(() => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const startOfWeek = new Date(today);
    startOfWeek.setDate(today.getDate() - today.getDay()); // Start of current week (Sunday)
    
    const endOfWeek = new Date(startOfWeek);
    endOfWeek.setDate(startOfWeek.getDate() + 6); // End of current week (Saturday)
    
    const startOfMonth = new Date(today.getFullYear(), today.getMonth(), 1);
    const endOfMonth = new Date(today.getFullYear(), today.getMonth() + 1, 0);
    
    return { today, startOfWeek, endOfWeek, startOfMonth, endOfMonth };
  }, []);

  // Optimized filter function
  const filterSession = useCallback((session: Session) => {
    // Filter by search query
    if (searchQuery) {
      const searchQueryLower = searchQuery.toLowerCase();
      const matchesSearch = 
        session.title?.toLowerCase().includes(searchQueryLower) || 
        session.program?.toLowerCase().includes(searchQueryLower) ||
        session.location?.toLowerCase().includes(searchQueryLower) ||
        session.instructor?.toLowerCase().includes(searchQueryLower) ||
        session.description?.toLowerCase().includes(searchQueryLower);
        
      if (!matchesSearch) return false;
    }
    
    // Filter by session type
    if (typeFilter !== 'all' && session.type !== typeFilter) {
      return false;
    }
    
    // Filter by program
    if (programFilter !== 'all' && session.program !== programFilter) {
      return false;
    }
    
    // Skip date filtering if showing all sessions
    if (dateFilter === 'all') return true;
    
    // Filter by date range
    const sessionDate = new Date(session.date);
    sessionDate.setHours(0, 0, 0, 0);
    const sessionTime = sessionDate.getTime();
    
    const todayTime = dateRanges.today.getTime();
    
    switch (dateFilter) {
      case 'today':
        return sessionTime === todayTime;
        
      case 'week':
        return sessionTime >= dateRanges.startOfWeek.getTime() && 
               sessionTime <= dateRanges.endOfWeek.getTime();
        
      case 'month':
        return sessionTime >= dateRanges.startOfMonth.getTime() && 
               sessionTime <= dateRanges.endOfMonth.getTime();
        
      case 'upcoming':
        // Only show sessions after today (not including today)
        return sessionTime > todayTime;
        
      case 'past':
        return sessionTime < todayTime;
        
      default:
        return true;
    }
  }, [searchQuery, typeFilter, programFilter, dateFilter, dateRanges]);

  // Apply filters to sessions
  const filteredSessions = useMemo(() => {
    if (!sessions.length) return [];
    
    // Only filter if we have active filters
    const hasActiveFilters = 
      searchQuery || 
      typeFilter !== 'all' || 
      programFilter !== 'all' || 
      dateFilter !== 'all';
    
    if (!hasActiveFilters) return sessions;
    
    return sessions.filter(filterSession);
  }, [sessions, filterSession, searchQuery, typeFilter, programFilter, dateFilter]);
  
  // Paginated sessions
  const paginatedSessions = useMemo(() => {
    const startIndex = (pagination.currentPage - 1) * pagination.pageSize;
    const endIndex = startIndex + pagination.pageSize;
    return filteredSessions.slice(startIndex, endIndex);
  }, [filteredSessions, pagination.currentPage, pagination.pageSize]);

  // Calculate session statistics
  const { sessionStats, weeklySessionCounts } = useMemo(() => {
    const stats = {
      total: 0,
      classes: 0,
      events: 0,
      activities: 0,
      totalStudents: 0
    };
    
    filteredSessions.forEach(session => {
      if (!session) return;
      stats.total++;
      if (session.type === 'class') stats.classes++;
      else if (session.type === 'event') stats.events++;
      else if (session.type === 'other') stats.activities++;
      stats.totalStudents += Number(session.students) || 0;
    });
    
    // Calculate weekly session counts
    const weeklyCounts = weekDates.map(day => {
      const dayStr = formatDateString(day);
      return filteredSessions.filter(s => s && s.date === dayStr).length;
    });
    
    return { sessionStats: stats, weeklySessionCounts: weeklyCounts };
  }, [filteredSessions, weekDates]);
  
  // Get sessions for the selected date
  const sessionsForSelectedDate = useMemo(() => {
    return filteredSessions.filter(session => 
      session && session.date === formatDateString(currentDate)
    );
  }, [filteredSessions, currentDate]);

  // Helper function to map SessionData to Session with proper date handling
  const mapToSession = (data: Partial<SessionData>, id?: number): Session => {
    // Ensure the date is in the correct format and timezone
    let sessionDate = data.date;
    if (sessionDate) {
      // If we have a date string, parse it and reformat it to ensure consistency
      const date = new Date(sessionDate);
      // Check if the date is valid
      if (!isNaN(date.getTime())) {
        // Format as YYYY-MM-DD without timezone conversion
        sessionDate = formatDateString(date);
      }
    } else {
      // If no date provided, use today's date in local timezone
      const today = new Date();
      sessionDate = formatDateString(today);
    }

    return {
      id: id || Date.now(),
      title: data.title || 'Untitled Session',
      type: data.attendanceType || 'class',
      time: data.timeIn && data.timeOut ? `${data.timeIn} - ${data.timeOut}` : 'TBD',
      time_in: data.timeIn || '',
      time_out: data.timeOut || '',
      students: 0,
      program: data.program || 'General',
      year: data.year || 'All Year Levels',
      section: data.section || 'All Sections',
      description: data.description || '',
      capacity: data.capacity || 'Unlimited',
      date: sessionDate
    } as Session;
  };

  // Function to handle adding/editing a session with proper date handling
  const handleSaveSession = async (sessionData: SessionData) => {
    try {
      // Ensure we have all required fields
      if (!sessionData.title || !sessionData.date || !sessionData.timeIn || !sessionData.timeOut || !sessionData.attendanceType) {
        throw new Error('Missing required fields');
      }

      // Format the date to ensure consistency
      const formattedDate = formatDateString(sessionData.date);
      
      // Prepare the session data for Supabase with proper types
      const sessionForSupabase = {
        title: sessionData.title,
        type: sessionData.attendanceType as 'class' | 'event' | 'other',
        time_in: sessionData.timeIn,
        time_out: sessionData.timeOut,
        program: sessionData.program || 'General',
        year: sessionData.year || 'All Year Levels',
        section: sessionData.section || 'All Sections',
        description: sessionData.description || '',
        capacity: sessionData.capacity ? String(sessionData.capacity) : 'Unlimited',
        date: formattedDate
      };

      console.log('Sending session data to server:', JSON.stringify(sessionForSupabase, null, 2));

      if (editingSession?.id) {
        // Update existing session
        const sessionId = editingSession.id;
        
        try {
          // Update the backend
          await updateSession(sessionId, { ...sessionForSupabase, capacity: parseInt(sessionForSupabase.capacity) || 0 });
          
          // Calculate student count for the updated session
          let studentCount = 0;
          try {
            // Build query to count students based on session criteria
            let countQuery = supabase
              .from('students')
              .select('*', { count: 'exact', head: true });
              
            // Apply filters only if they are not "all" values
            if (sessionForSupabase.program && !sessionForSupabase.program.toLowerCase().includes('all')) {
              countQuery = countQuery.eq('program', sessionForSupabase.program.trim());
            }
            
            if (sessionForSupabase.year && !sessionForSupabase.year.toLowerCase().includes('all')) {
              let yearValue = sessionForSupabase.year.trim();
              if (yearValue.endsWith(' Year')) {
                yearValue = yearValue.replace(' Year', '');
              }
              countQuery = countQuery.eq('year', yearValue);
            }
            
            if (sessionForSupabase.section && !sessionForSupabase.section.toLowerCase().includes('all')) {
              countQuery = countQuery.eq('section', sessionForSupabase.section.trim());
            }
            
            const { count } = await countQuery;
            studentCount = count || 0;
          } catch (countError) {
            console.error('Error calculating student count:', countError);
            studentCount = 0;
          }
          
          // Force a refresh of the sessions list
          setSessions(prevSessions => {
            return prevSessions.map(s => 
              s.id === sessionId 
                ? { 
                    ...s, 
                    ...sessionForSupabase,
                    time: `${sessionForSupabase.time_in} - ${sessionForSupabase.time_out}`,
                    students: studentCount // Use calculated student count
                  } 
                : s
            );
          });
          
          toast({
            title: "Success",
            description: "Session updated successfully",
            variant: "default"
          });
        } catch (updateError) {
          console.error('Error updating session:', updateError);
          throw updateError;
        }
      } else {
        // Add new session
        try {
          const newSession = await createSession({ ...sessionForSupabase, capacity: parseInt(sessionForSupabase.capacity) || 0 });
          
          // Calculate student count for the new session
          let studentCount = 0;
          try {
            // Build query to count students based on session criteria
            let countQuery = supabase
              .from('students')
              .select('*', { count: 'exact', head: true });
              
            // Apply filters only if they are not "all" values
            if (sessionForSupabase.program && !sessionForSupabase.program.toLowerCase().includes('all')) {
              countQuery = countQuery.eq('program', sessionForSupabase.program.trim());
            }
            
            if (sessionForSupabase.year && !sessionForSupabase.year.toLowerCase().includes('all')) {
              let yearValue = sessionForSupabase.year.trim();
              if (yearValue.endsWith(' Year')) {
                yearValue = yearValue.replace(' Year', '');
              }
              countQuery = countQuery.eq('year', yearValue);
            }
            
            if (sessionForSupabase.section && !sessionForSupabase.section.toLowerCase().includes('all')) {
              countQuery = countQuery.eq('section', sessionForSupabase.section.trim());
            }
            
            const { count } = await countQuery;
            studentCount = count || 0;
          } catch (countError) {
            console.error('Error calculating student count:', countError);
            studentCount = 0;
          }
          
          // Create the full session object with all required fields
          const fullNewSession: Session = {
            ...sessionForSupabase,
            id: newSession.id,
            time: `${sessionForSupabase.time_in} - ${sessionForSupabase.time_out}`,
            students: studentCount,
            created_at: newSession.created_at || new Date().toISOString(),
            location: '',
            instructor: '',
            updated_at: newSession.updated_at || new Date().toISOString(),
            created_by_user_id: newSession.created_by_user_id
          };
          
          // Update the state with the new session - no need to call loadSessions()
          // as we already have all the data we need in fullNewSession
          setSessions(prevSessions => {
            // Check if session already exists to prevent duplicates
            if (prevSessions.some(s => s.id === fullNewSession.id)) {
              return prevSessions;
            }
            return [...prevSessions, fullNewSession];
          });
          
          toast({
            title: "Success",
            description: "Session created successfully",
            variant: "default"
          });
        } catch (createError) {
          console.error('Error creating session:', createError);
          throw createError;
        }
      }
      
      // Reset the form and close the modal
      setEditingSession(null);
      setIsModalOpen(false);
      
    } catch (error) {
      console.error('Error saving session:', error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to save session. Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleEditSession = async (session: Session) => {
    try {
      // Convert 12-hour format to 24-hour format for time inputs
      const convertTo24Hour = (timeStr: string) => {
        if (!timeStr) return '';
        
        // If already in 24-hour format (HH:MM), return as is
        if (/^([01]?[0-9]|2[0-3]):[0-5][0-9]$/.test(timeStr)) {
          return timeStr;
        }
        
        // Handle 12-hour format with AM/PM
        const match = timeStr.match(/(\d+):(\d+)\s*(AM|PM)/i);
        if (!match) return '';
        
        const [_, hours, minutes, period] = match;
        let hh = parseInt(hours, 10);
        
        if (period.toUpperCase() === 'PM' && hh < 12) {
          hh += 12;
        } else if (period.toUpperCase() === 'AM' && hh === 12) {
          hh = 0;
        }
        
        return `${String(hh).padStart(2, '0')}:${minutes}`;
      };
  
      // Handle time splitting more robustly
      let timeIn = '';
      let timeOut = '';
      
      if (session.time) {
        const timeParts = session.time.split(' - ');
        timeIn = convertTo24Hour(timeParts[0] || '');
        timeOut = convertTo24Hour(timeParts[1] || '');
      } else {
        // Fallback to direct time fields if time string is not available
        timeIn = convertTo24Hour(session.time_in || '');
        timeOut = convertTo24Hour(session.time_out || '');
      }
  
      // Debug log to check the session data
      console.log('Editing session:', {
        ...session,
        timeIn,
        timeOut,
        year: session.year
      });
      
      const sessionData: SessionDataWithId = {
        id: session.id,
        title: session.title,
        program: session.program || '',
        // Ensure we handle the year properly
        year: session.year || 'All Year Levels',
        section: session.section || '',
        date: session.date,
        timeIn,
        timeOut,
        description: session.description || '',
        venue: session.location || 'Not specified',
        capacity: session.capacity || 'Unlimited',
        attendanceType: (session.type as 'class' | 'event' | 'other') || 'class'
      };
      
      setEditingSession(sessionData);
      setIsModalOpen(true);
    } catch (error) {
      console.error('Error preparing session for edit:', error);
      toast({
        title: "Error",
        description: "Failed to prepare session for editing. Please try again.",
        variant: "destructive"
      });
    }
  };

  // Function to confirm session deletion
  const confirmDeleteSession = (sessionId: number) => {
    setSessionToDelete(sessionId);
    setIsDeleteDialogOpen(true);
  };

  // Function to handle deleting a session
  const handleDeleteSession = async () => {
    if (!sessionToDelete) return;
    
    try {
      await deleteSession(sessionToDelete);
      setSessions(sessions.filter(session => session.id !== sessionToDelete));
      toast({
        title: "Success",
        description: "Session deleted successfully.",
      });
    } catch (error) {
      console.error('Error deleting session:', error);
      toast({
        title: "Error",
        description: "Failed to delete session. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsDeleteDialogOpen(false);
      setSessionToDelete(null);
    }
  };

  // Helper function to get icon for session type
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'class':
        return <BookOpen className="w-4 h-4 text-primary" />;
      case 'event':
        return <Calendar className="w-4 h-4 text-accent" />;
      case 'other':
        return <Star className="w-4 h-4 text-education-navy" />;
      default:
        return <BookOpen className="w-4 h-4 text-primary" />;
    }
  };

  // Helper function to get badge for session type
  const getTypeBadge = (type: string) => {
    switch (type) {
      case 'class':
        return <Badge className="bg-primary/10 text-primary border-primary/20">Class</Badge>;
      case 'event':
        return <Badge className="bg-accent/10 text-accent border-accent/20">Event</Badge>;
      case 'other':
        return <Badge className="bg-education-navy/10 text-education-navy border-education-navy/20">Activity</Badge>;
      default:
        return <Badge>Unknown</Badge>;
    }
  };

  // Function to navigate between weeks
  const navigateDate = useCallback((direction: 'prev' | 'next' | 'today') => {
    setCurrentDate(prevDate => {
      const newDate = new Date(prevDate);
      
      if (direction === 'today') {
        // Reset to today's date
        const today = new Date();
        return new Date(today.getFullYear(), today.getMonth(), today.getDate());
      }
      
      // Move by 7 days for previous/next week
      const daysToAdd = direction === 'next' ? 7 : -7;
      newDate.setDate(prevDate.getDate() + daysToAdd);
      
      console.log('Navigating to week starting from:', newDate.toISOString().split('T')[0]);
      return newDate;
    });
  }, []);

  // Format date for display
  const formatDate = useCallback((date: Date) => {
    return date.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  }, []);

  // No need for a full page loading state anymore
  // The loading indicator is now shown in the sessions list section

  // Handle error state
  if (error) {
    return (
      <Layout>
        <div className="p-8">
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative" role="alert">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        </div>
      </Layout>
    );
  }
  
  if (error) {
    return (
      <Layout>
        <div className="p-8">
          <div className="bg-red-50 border-l-4 border-red-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <Dialog open={isModalOpen} onOpenChange={(open) => {
        setIsModalOpen(open);
        if (!open) {
          setEditingSession(null);
          // Clear any form state that might persist
          setTimeout(() => {
            setEditingSession(null);
          }, 100);
        }
      }}>
        <DialogContent className="max-w-4xl max-h-[90vh]">
          
          <div className="py-4">
            <AttendanceForm 
              onSuccess={() => {
                setIsModalOpen(false);
                setEditingSession(null);
              }} 
              onSubmit={handleSaveSession} 
              initialData={editingSession}
            />
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle className="text-education-navy">Delete Session</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <p>Are you sure you want to delete this session? This action cannot be undone.</p>
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setIsDeleteDialogOpen(false)}
              className="text-education-navy"
            >
              Cancel
            </Button>
            <Button 
              variant="destructive"
              onClick={handleDeleteSession}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete Session
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4">
          <div className="mb-3">
            <div>
              <h1 className="text-2xl font-bold text-education-navy">SESSIONS</h1>
            </div>
          </div>
          
          {/* Big space between page title and card */}
          <div className="mb-16"></div>
          
          {/* Sessions Section */}
          <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-semibold text-education-navy">List of Sessions {pagination.currentPage > 1 && `(Page ${pagination.currentPage})`}</h3>
              <Button
                variant="default"
                size="sm"
                className="h-8"
                onClick={() => setIsModalOpen(true)}
              >
                <Plus className="w-4 h-4 mr-1" />
                Add Session
              </Button>
            </div>
            
            {/* Big space below List of Sessions label */}
            <div className="mb-8"></div>
            
            {/* Controls */}
            <div className="flex items-center justify-between mb-4 p-0">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">Showed:</span>
                <Select value={displayPageSize === totalSessionsCount ? 'all' : displayPageSize.toString()} onValueChange={handlePageSizeChange}>
                  <SelectTrigger className="w-20 h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="10">10</SelectItem>
                    <SelectItem value="100">100</SelectItem>
                    <SelectItem value="250">250</SelectItem>
                    <SelectItem value="all">ALL</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">Search:</span>
                <Input
                  placeholder="Search sessions..."
                  className="w-64 h-8 text-sm"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {/* Sessions List */}
            <div className="space-y-2">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-education-blue mr-2" />
                  <span className="text-sm text-muted-foreground">Loading sessions...</span>
                </div>
              ) : paginatedSessions.length > 0 ? (
                [...paginatedSessions].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()).map((session) => (
                <Card key={session.id} className="bg-gradient-card border border-gray-200 shadow-card">
                  <CardContent className="p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-lg bg-gradient-primary/10">
                          {getTypeIcon(session.type)}
                        </div>
                        <div>
                          <div className="flex items-center gap-1.5 mb-0.5">
                            <h4 className="font-medium text-education-navy text-sm">{session.title}</h4>
                            {getTypeBadge(session.type)}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            <span className="mr-3">{format(new Date(session.date), 'MMM d, yyyy')}</span>
                            <span className="inline-flex items-center gap-1"><Clock className="w-2.5 h-2.5" />{session.time}</span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-0.5">
                            {session.program} • {session.year}
                            {session.section && ` • ${session.section}`}
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-6">
                        <div className="hidden sm:flex flex-wrap items-center gap-6">
                          <div className="text-center">
                            <div className="flex items-center gap-1 text-xs text-muted-foreground">
                              <Users className="w-3.5 h-3.5" />
                              <span>Students</span>
                            </div>
                            <div className="text-sm font-bold text-education-navy">{session.students}</div>
                          </div>
                          <div className="text-center">
                            <div className="text-xs text-muted-foreground">Present</div>
                            <div className="text-sm font-bold text-accent">{session.present ?? 0}</div>
                          </div>
                          <div className="text-center">
                            <div className="text-xs text-muted-foreground">Absent</div>
                            <div className="text-sm font-bold text-destructive">{session.absent ?? 0}</div>
                          </div>
                        </div>

                        <div className="flex gap-1">
                          <Button 
                            variant="outline" 
                            size="sm"
                            className="h-8 w-8 p-0"
                            onClick={() => navigate(`/sessions/${session.id}/students`)}
                            title="View Students"
                          >
                            <Users className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="outline"
                            size="sm"
                            className="h-8 w-8 p-0"
                            onClick={() => handleEditSession(session)}
                            title="Edit Session"
                          >
                            <SquarePen className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="outline"
                            size="sm"
                            className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                            onClick={() => confirmDeleteSession(session.id)}
                            title="Delete Session"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            ) : (
              <Card className="bg-gradient-card border border-gray-200 shadow-card">
                <CardContent className="p-8 text-center">
                  <CalendarClock className="w-10 h-10 mx-auto text-muted-foreground mb-2" />
                  <h4 className="text-lg font-medium text-education-navy">
                    {sessions.length === 0 ? 'No sessions scheduled' : 'No matching sessions found'}
                  </h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    {sessions.length === 0 
                      ? 'There are no scheduled sessions.' 
                      : 'Try adjusting your search or filters.'}
                  </p>
                  <Button 
                    variant="outline" 
                    className="mt-4"
                    onClick={() => setIsModalOpen(true)}
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Session
                  </Button>
                </CardContent>
              </Card>
              )}
            </div>
          </div>
        </div>
      </PageWrapper>
    </Layout>
  );
};

export default Schedule;