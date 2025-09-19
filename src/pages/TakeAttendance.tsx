import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardFooter, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { 
  Loader2, 
  Clock, 
  MapPin, 
  Calendar as CalendarIcon,
  CalendarDays, 
  List, 
  Star,
  RefreshCw,
  BookOpen,
  Users,
  CalendarClock
} from "lucide-react";
import { format, isToday, parseISO, isBefore, isAfter, isSameDay } from "date-fns";
import { DateRange } from "react-day-picker";
import { useNavigate, useSearchParams } from "react-router-dom";
import { supabase } from "@/lib/supabase";
import { useEffect, useState, useCallback } from "react";
import Layout from "@/components/Layout";

// Format date as 'Month Day, Year' (e.g., 'January 1, 2023')
const formatDate = (dateString: string): string => {
  return format(parseISO(dateString), 'MMMM d, yyyy');
};

// Format time as 'HH:MM AM/PM' (e.g., '02:30 PM')
const formatTime = (timeString: string): string => {
  if (!timeString) return '--:--';
  return format(parseISO(`2000-01-01T${timeString}`), 'h:mm a');
};

type SessionStatus = 'upcoming' | 'in-progress' | 'completed';

type Session = {
  id: number;
  created_at: string;
  updated_at: string;
  title: string;
  description: string;
  type: 'class' | 'event' | 'other';
  date: string;
  time_in: string;
  time_out: string;
  created_by_user_id?: string;
  capacity: number;
  program: string;
  year: string;
  section: string;
  enrolled_students?: Array<{
    id: string;
    name: string;
    email: string;
    status: 'present' | 'absent' | 'late' | 'excused';
  }>;
  attendees_count?: number;
  creator?: {
    first_name: string;
    last_name: string;
    role: string;
  };
};

const ITEMS_PER_PAGE = 10;

// Get session status based on current time
const getSessionStatus = (session: Session): SessionStatus => {
  const now = new Date();
  const sessionDate = new Date(session.date);
  const startTime = new Date(session.date + 'T' + session.time_in);
  const endTime = new Date(session.date + 'T' + session.time_out);
  
  if (now < startTime) return 'upcoming';
  if (now >= startTime && now <= endTime) return 'in-progress';
  return 'completed';
};

// Get status badge component
const getStatusBadge = (status: SessionStatus) => {
  switch (status) {
    case 'upcoming':
      return <Badge variant="outline" className="bg-yellow-50 text-yellow-800 border-yellow-200">Upcoming</Badge>;
    case 'in-progress':
      return <Badge variant="outline" className="bg-green-50 text-green-800 border-green-200">In Progress</Badge>;
    case 'completed':
      return <Badge variant="outline" className="bg-gray-100 text-gray-800 border-gray-200">Completed</Badge>;
    default:
      return null;
  }
};

const TakeAttendanceContent: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Status filter removed as per user request
  // Remove date range state as we'll fetch all sessions
  const [dateRange] = useState<DateRange | undefined>(undefined);
  const [currentPage, setCurrentPage] = useState(1);
  const [sessionTypeFilter, setSessionTypeFilter] = useState<string>('all');
  const [filteredSessions, setFilteredSessions] = useState<Session[]>([]);

  // Fetch sessions from Supabase
  const fetchSessions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Build the query with filters
      let query = supabase
        .from('sessions')
        .select(`*`, { count: 'exact' });
      
      // Removed date range filter to fetch all sessions
      // Date filtering will be handled client-side for the tabs
      
      // Apply session type filter
      if (sessionTypeFilter !== 'all') {
        query = query.eq('type', sessionTypeFilter);
      }
      
      // Apply sorting
      query = query
        .order('date', { ascending: true })
        .order('time_in', { ascending: true });
      
      // Execute the query
      const { data, error, count } = await query;
      
      if (error) {
        throw error;
      }
      
      // Resolve creators from admin/users
      const creatorIds = Array.from(new Set((data || [])
        .map((s: any) => s.created_by_user_id)
        .filter((v: any) => !!v)));

      const creatorsMap = new Map<string, { first_name: string; last_name: string; role: string }>();
      if (creatorIds.length > 0) {
        const { data: adminRows } = await supabase
          .from('admin')
          .select('id, first_name, last_name')
          .in('id', creatorIds as any);
        (adminRows || []).forEach((r: any) => {
          creatorsMap.set(r.id, { first_name: r.first_name, last_name: r.last_name, role: 'admin' });
        });
        const { data: userRows } = await supabase
          .from('users')
          .select('id, first_name, last_name, role')
          .in('id', creatorIds as any);
        (userRows || []).forEach((r: any) => {
          creatorsMap.set(r.id, { first_name: r.first_name, last_name: r.last_name, role: r.role || 'user' });
        });
      }

      // Transform the data
      const formattedSessions: Session[] = (data || []).map(session => ({
        id: session.id,
        created_at: session.created_at || new Date().toISOString(),
        updated_at: session.updated_at || new Date().toISOString(),
        title: session.title || 'Untitled Session',
        description: session.description || '',
        type: (session.type as 'class' | 'event' | 'other') || 'class',
        location: '',
        instructor: '',
        date: session.date,
        time_in: session.time_in || '00:00',
        time_out: session.time_out || '00:00',
        created_by_user_id: session.created_by_user_id,
        creator: session.created_by_user_id ? creatorsMap.get(session.created_by_user_id) : undefined,
        capacity: parseInt(session.capacity) || 0,
        program: session.program || '',
        year: session.year || '',
        section: session.section || ''
      }));
      
      setSessions(formattedSessions);
      return formattedSessions;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      console.error('Error details:', {
        message: errorMessage,
        name: err?.name,
        stack: err?.stack,
        date: new Date().toISOString()
      });
      setError(`Failed to load sessions: ${errorMessage}`);
      return [];
    } finally {
      setLoading(false);
    }
  }, [dateRange, sessionTypeFilter, setError, setLoading, setSessions]);

  // Load sessions on component mount and when filters change
  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Filter sessions based on session type only
  useEffect(() => {
    const filtered = sessions.filter(session => {
      // Apply session type filter
      return sessionTypeFilter === 'all' || session.type === sessionTypeFilter;
    });
    
    setFilteredSessions(filtered);
    setCurrentPage(1); // Reset to first page when filter changes
  }, [sessions, sessionTypeFilter, setFilteredSessions, setCurrentPage]);


  // Categorize sessions
  const now = new Date();
  const todayStart = new Date(now);
  todayStart.setHours(0, 0, 0, 0);
  
  const todaysSessions = filteredSessions.filter(session => {
    const sessionDate = new Date(session.date);
    return isSameDay(sessionDate, now);
  });
  

  
  const pastSessions = filteredSessions.filter(session => {
    const sessionDate = new Date(session.date);
    return isBefore(sessionDate, todayStart);
  });
  const handleStartAttendance = useCallback((sessionId: number) => {
    navigate(`/take-attendance/${sessionId}`);
  }, [navigate]);

  const renderSessionList = (sessions: Session[]) => {
    if (sessions.length === 0) {
      return (
        <div className="text-center py-8 text-muted-foreground">
          No sessions found
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {sessions.map((session) => (
          <SessionCard
            key={session.id}
            session={session}
            onStartAttendance={handleStartAttendance}
          />
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">Error: </strong>
        <span className="block sm:inline">{error}</span>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col space-y-4 px-6 py-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-education-navy">TAKE ATTENDANCE</h1>
          <p className="text-sm text-muted-foreground">
            Select a session to take attendance
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchSessions}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Space between title and content to match other pages */}
      <div className="mb-16"></div>

      <div className="flex items-center space-x-4">
        {/* Search removed as per user request */}
      </div>

      <Tabs defaultValue="today" className="space-y-4">
        <TabsList>
          <TabsTrigger value="today" className="relative">
            Today
            {todaysSessions.length > 0 && (
              <Badge className="ml-2 h-5 w-5 rounded-full bg-primary text-primary-foreground text-xs flex items-center justify-center">
                {todaysSessions.length}
              </Badge>
            )}
          </TabsTrigger>

          <TabsTrigger value="past" className="relative">
            Past
            {pastSessions.length > 0 && (
              <Badge className="ml-2 h-5 w-5 rounded-full bg-primary text-primary-foreground text-xs flex items-center justify-center">
                {pastSessions.length}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="today" className="space-y-4">
          {todaysSessions.length > 0 ? (
            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {todaysSessions.map((session) => (
                <SessionCard
                  key={session.id}
                  session={session}
                  onStartAttendance={handleStartAttendance}
                />
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <CalendarClock className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No sessions today</h3>
              <p className="text-sm text-muted-foreground">
                There are no sessions scheduled for today.
              </p>
            </div>
          )}
        </TabsContent>



        <TabsContent value="past" className="space-y-4">
          {pastSessions.length > 0 ? (
            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {pastSessions.map((session) => (
                <SessionCard
                  key={session.id}
                  session={session}
                  onStartAttendance={handleStartAttendance}
                />
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <CalendarClock className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium">No past sessions</h3>
              <p className="text-sm text-muted-foreground">
                There are no past sessions to display.
              </p>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

// SessionCard component to render individual session cards
interface SessionCardProps {
  session: Session;
  onStartAttendance: (id: number) => void;
}

const SessionCard: React.FC<SessionCardProps> = ({ session, onStartAttendance }) => {
  return (
    <Card className="relative overflow-hidden shadow-sm">
      <div className="absolute top-2 right-2">
        {session.type === 'class' && (
          <Badge className="bg-primary/10 text-primary border-primary/20 text-xs">Class</Badge>
        )}
        {session.type === 'event' && (
          <Badge className="bg-accent/10 text-accent border-accent/20 text-xs">Event</Badge>
        )}
        {session.type === 'other' && (
          <Badge className="bg-education-navy/10 text-education-navy border-education-navy/20 text-xs">Activity</Badge>
        )}
      </div>
      <CardHeader className="p-3 pb-1">
        <div className="flex items-center gap-2">
          {session.type === 'class' && <BookOpen className="h-4 w-4 text-primary" />}
          {session.type === 'event' && <CalendarIcon className="h-4 w-4 text-accent" />}
          {session.type === 'other' && <Star className="h-4 w-4 text-education-navy" />}
          <CardTitle className="text-base">{session.title}</CardTitle>
        </div>
        <div className="text-xs text-muted-foreground mt-0.5">
          {session.program} • {session.year} {session.section && `• ${session.section}`}
        </div>
      </CardHeader>
      <CardContent className="p-3 pt-0">
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="space-y-1">
            <div className="flex items-center text-muted-foreground">
              <CalendarIcon className="h-3.5 w-3.5 mr-1.5" />
              <span>{formatDate(session.date)}</span>
            </div>
            <div className="flex items-center text-muted-foreground">
              <Clock className="h-3.5 w-3.5 mr-1.5" />
              <span>{formatTime(session.time_in)} - {formatTime(session.time_out)}</span>
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-muted-foreground text-ellipsis overflow-hidden">
              <span className="font-medium">Created by:</span> {
                session.creator
                  ? `${session.creator.role} (${session.creator.first_name} ${session.creator.last_name})`
                  : 'System'
              }
            </div>
          </div>
        </div>
        <Button 
          size="sm"
          className="w-full mt-3 text-xs h-8 bg-teal-300 text-teal-900 hover:bg-teal-200 shadow-glow hover:shadow-elegant transition-all duration-200"
          onClick={() => onStartAttendance(session.id)}
        >
          Start Attendance
        </Button>
      </CardContent>
    </Card>
  );
};

const TakeAttendance: React.FC = () => {
  return (
    <Layout>
      <TakeAttendanceContent />
    </Layout>
  );
};

export default TakeAttendance;