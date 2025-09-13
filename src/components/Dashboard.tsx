import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CalendarDays, Users, UserCheck, BarChart3, CalendarClock, CheckCircle, TrendingUp, TrendingDown, Activity } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { supabase } from "@/lib/supabase";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchUserRole } from "@/lib/getUserRole";

// Use the same role caching system as navigation
const getCachedUserRole = (): string | null => {
  try {
    return localStorage.getItem('userRole');
  } catch {
    return null;
  }
};

const getCachedUserId = (): string | null => {
  try {
    return localStorage.getItem('userId');
  } catch {
    return null;
  }
};

const setCachedUserRole = (role: string, userId: string) => {
  try {
    localStorage.setItem('userRole', role);
    localStorage.setItem('userId', userId);
  } catch {
    // Ignore localStorage errors
  }
};

let cachedUserRole: string | null = getCachedUserRole();
let cachedUserId: string | null = getCachedUserId();

// Enhanced mock data generation with more realistic patterns
const generateMockData = (period: 'daily' | 'weekly' | 'monthly') => {
  if (period === 'daily') {
    return Array.from({ length: 7 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - i));
      const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
      const isWeekend = date.getDay() === 0 || date.getDay() === 6;
      const baseAttendance = isWeekend ? 0.85 : 0.92; // Lower attendance on weekends
      const totalStudents = 1500;
      const present = Math.floor(totalStudents * baseAttendance + (Math.random() - 0.5) * 50);
      const absent = totalStudents - present;
      return {
        name: dayName,
        present,
        absent,
        date: date.toISOString().split('T')[0]
      };
    });
  } else if (period === 'weekly') {
    return Array.from({ length: 8 }, (_, i) => {
      const baseAttendance = 0.90 + (Math.random() - 0.5) * 0.1;
      const totalStudents = 1500;
      const present = Math.floor(totalStudents * baseAttendance);
      const absent = totalStudents - present;
      return {
        name: `Week ${i + 1}`,
        present,
        absent,
        week: i + 1
      };
    });
  } else { // monthly
    return [
      { name: 'Jan', present: 1350, absent: 150, month: 1 },
      { name: 'Feb', present: 1320, absent: 180, month: 2 },
      { name: 'Mar', present: 1380, absent: 120, month: 3 },
      { name: 'Apr', present: 1410, absent: 90, month: 4 },
      { name: 'May', present: 1440, absent: 60, month: 5 },
      { name: 'Jun', present: 1470, absent: 30, month: 6 },
      { name: 'Jul', present: 1485, absent: 15, month: 7 },
      { name: 'Aug', present: 1500, absent: 0, month: 8 },
    ];
  }
};



const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [userProfile, setUserProfile] = useState<any>(null);
  const [userRole, setUserRole] = useState<string>(() => {
    return cachedUserRole || 'user';
  });
  const [totalStudents, setTotalStudents] = useState(0);
  const [loading, setLoading] = useState(true);
  const isInitialMount = useRef(true);
  const [timePeriod, setTimePeriod] = useState<'daily' | 'weekly' | 'monthly'>('monthly');
  const [chartData, setChartData] = useState<Array<{name: string, present: number, absent: number}>>([]);
  const [realTimeStats, setRealTimeStats] = useState({
    todayAttendance: 0,
    todaySessions: 0,
    activeClasses: 0,
    pendingExcuses: 0
  });

  // Mock placeholders for new dashboard subtitles (replace with real data later)
  const mockYesterdayPresents = 1354; // Count of presents yesterday
  const mockYesterdayAttendanceRate = 93.8; // % yesterday
  const mockCompletedSessionsToday = 13; // completed sessions today

  const yesterdayHigherAttendance = mockYesterdayPresents > realTimeStats.todayAttendance;
  const yesterdayHigherRate = mockYesterdayAttendanceRate > 94.2; // today's mock rate
  
  // Academic Year state
  const [academicYear, setAcademicYear] = useState<{
    year: string;
    semester: string;
    status: 'active' | 'inactive';
  } | null>(null);
  
  // Recent Sessions interface and state
  interface RecentSession {
    id: string;
    course: string;
    students: number;
    time: string;
    timeAgo: string;
    status: 'completed' | 'ongoing' | 'upcoming';
    attendanceRate?: number;
  }
  
  const [recentSessions, setRecentSessions] = useState<RecentSession[]>([]);

  // Enhanced fetch recent sessions with more realistic data
  const fetchRecentSessions = useCallback(async () => {
    try {
      const mockSessions: RecentSession[] = [
        {
          id: '1',
          course: 'CS 101 - Introduction to Programming',
          students: 45,
          time: '9:00 AM - 10:30 AM',
          timeAgo: '2 min ago',
          status: 'completed',
          attendanceRate: 93
        },
        {
          id: '2',
          course: 'MATH 201 - Calculus II',
          students: 38,
          time: '11:00 AM - 12:30 PM',
          timeAgo: '15 min ago',
          status: 'ongoing',
          attendanceRate: 87
        },
        {
          id: '3',
          course: 'ENG 101 - English Composition',
          students: 42,
          time: '2:00 PM - 3:30 PM',
          timeAgo: '1 hour ago',
          status: 'upcoming'
        },
        {
          id: '4',
          course: 'PHY 101 - Physics Fundamentals',
          students: 35,
          time: '4:00 PM - 5:30 PM',
          timeAgo: '3 hours ago',
          status: 'completed',
          attendanceRate: 91
        },
        {
          id: '5',
          course: 'CHEM 101 - General Chemistry',
          students: 40,
          time: '8:00 AM - 9:30 AM',
          timeAgo: '5 hours ago',
          status: 'completed',
          attendanceRate: 88
        }
      ];
      
      setRecentSessions(mockSessions);
    } catch (error) {
      console.error('Error fetching recent sessions:', error);
    }
  }, []);

  // Fetch real-time statistics
  const fetchRealTimeStats = useCallback(async () => {
    try {
      // Mock real-time data - in production, this would fetch from actual database
      setRealTimeStats({
        todayAttendance: 1410,
        todaySessions: 8,
        activeClasses: 24,
        pendingExcuses: 12
      });
    } catch (error) {
      console.error('Error fetching real-time stats:', error);
    }
  }, []);

  // Fetch current academic year
  const fetchAcademicYear = useCallback(async () => {
    try {
      // Mock academic year data - in production, this would fetch from actual database
      setAcademicYear({
        year: '2024-2025',
        semester: 'First Semester',
        status: 'active'
      });
    } catch (error) {
      console.error('Error fetching academic year:', error);
    }
  }, []);

  useEffect(() => {
    loadUser();
    fetchTotalStudents();
    fetchRecentSessions();
    fetchRealTimeStats();
    fetchAcademicYear();
    
    // Initialize chart data
    setChartData(generateMockData(timePeriod));
  }, [user, timePeriod]);

  const loadUser = async () => {
    // If we have cached role for the same user, don't refetch
    if (cachedUserRole && cachedUserId === user?.id) {
      setUserRole(cachedUserRole);
      // Still fetch profile for other data, but don't wait for it
      fetchProfileData();
      return;
    }

    if (!user) {
      const defaultRole = 'user';
      setUserRole(defaultRole);
      cachedUserRole = defaultRole;
      cachedUserId = null;
      return;
    }
    
    try {
      // Use new helper to resolve role from admin/users
      const role = await fetchUserRole(user.id);
      setUserRole(role);
      cachedUserRole = role;
      cachedUserId = user.id;
      setCachedUserRole(role, user.id);
      // fetch profile-like data from admin/users for display
      await fetchProfileData();
    } catch (error) {
      console.error('Error resolving user role:', error);
      const defaultRole = 'user';
      setUserRole(defaultRole);
      if (user?.id) {
        cachedUserRole = defaultRole;
        cachedUserId = user.id;
        setCachedUserRole(defaultRole, user.id);
      }
    }
  };

  const fetchProfileData = async () => {
    if (!user) return;
    try {
      // Try admin first
      let profile: any = null;
      const { data: adminData } = await supabase
        .from('admin')
        .select('id, email, first_name, last_name, status, created_at, updated_at')
        .eq('id', user.id)
        .maybeSingle();
      if (adminData) profile = adminData;
      if (!profile) {
        const { data: userData } = await supabase
          .from('users')
          .select('id, email, role, first_name, last_name, status, created_at, updated_at')
          .eq('id', user.id)
          .maybeSingle();
        if (userData) profile = userData;
      }
      setUserProfile(profile);
    } catch (error) {
      console.error('Error fetching account profile:', error);
    }
  };

  const fetchTotalStudents = async () => {
    try {
      const { count, error } = await supabase
        .from('students')
        .select('*', { count: 'exact', head: true });
      
      if (error) throw error;
      setTotalStudents(count || 0);
    } catch (error) {
      console.error('Error fetching total students:', error);
    } finally {
      setLoading(false);
    }
  };

  // Handle session click
  const handleSessionClick = (session: RecentSession) => {
    if (session.status === 'ongoing') {
      navigate('/take-attendance');
    } else if (session.status === 'completed') {
              navigate('/reports');
    } else {
      navigate('/schedule');
    }
  };



  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning";
    if (hour < 18) return "Good afternoon";
    return "Good evening";
  };

  const getDashboardTitle = () => {
    const roleLabels = {
      admin: 'ADMIN',
      'ROTC admin': 'ROTC ADMIN',
      Instructor: 'INSTRUCTOR',
      'SSG officer': 'SSG OFFICER',
      'ROTC officer': 'ROTC OFFICER'
    };
    return `${roleLabels[userRole as keyof typeof roleLabels] || 'User'} DASHBOARD`;
  };

  const getUserDisplayName = () => {
    // Don't show email while profile is still loading
    if (loading && !userProfile) {
      return '';
    }
    
    if (userProfile?.first_name && userProfile?.last_name) {
      return `${userProfile.first_name} ${userProfile.last_name}`;
    }
    if (userProfile?.first_name) {
      return userProfile.first_name;
    }
    if (user?.email) {
      return user.email.split('@')[0];
    }
    return 'User';
  };

  // Update chart data when time period changes
  useEffect(() => {
    setChartData(generateMockData(timePeriod));
  }, [timePeriod]);

  // Calculate percentages for the chart
  const chartDataWithPercentage = useMemo(() => {
    return chartData.map(item => ({
      ...item,
      presentPercentage: Math.round((item.present / (item.present + item.absent)) * 100),
      absentPercentage: Math.round((item.absent / (item.present + item.absent)) * 100)
    }));
  }, [chartData]);

  return (
    <div className="flex-1 space-y-4 px-6 py-4 opacity-100 transition-opacity duration-300">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold tracking-tight text-gray-900">{getDashboardTitle()}</h2>
          <p className="text-sm text-gray-600">
            {getGreeting()}! Here's your attendance overview.
          </p>
        </div>
      </div>
      
      {/* Minimalist Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Total Students</CardTitle>
            <div className="p-2 bg-blue-100 rounded-lg">
              <Users className="h-4 w-4 text-blue-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {loading ? '' : totalStudents.toLocaleString()}
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <BarChart3 className="h-3 w-3 mr-1" />
              +5.2% this month
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Today's Attendance</CardTitle>
            <div className="p-2 bg-purple-100 rounded-lg">
              <UserCheck className="h-4 w-4 text-purple-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {realTimeStats.todayAttendance.toLocaleString()}
            </div>
            <div className="flex items-center text-sm text-gray-600">
              {yesterdayHigherAttendance ? (
                <TrendingUp className="h-3 w-3 mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1" />
              )}
              {mockYesterdayPresents.toLocaleString()} yesterday
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Attendance Rate</CardTitle>
            <div className="p-2 bg-green-100 rounded-lg">
              <Activity className="h-4 w-4 text-green-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-3xl font-bold text-gray-900 mb-1">94.2%</div>
            <div className="flex items-center text-sm text-gray-600">
              {yesterdayHigherRate ? (
                <TrendingUp className="h-3 w-3 mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1" />
              )}
              {mockYesterdayAttendanceRate}% yesterday
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Today's Sessions</CardTitle>
            <div className="p-2 bg-orange-100 rounded-lg">
              <CalendarClock className="h-4 w-4 text-orange-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {realTimeStats.activeClasses}
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <CheckCircle className="h-3 w-3 mr-1" />
              {mockCompletedSessionsToday} completed
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Academic Year</CardTitle>
            <div className="p-2 bg-indigo-100 rounded-lg">
              <CalendarDays className="h-4 w-4 text-indigo-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-2xl font-bold text-gray-900 mb-1">
              {academicYear?.year || '2024-2025'}
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <CalendarDays className="h-3 w-3 mr-1" />
              {academicYear?.semester || 'First Semester'}
            </div>
          </CardContent>
        </Card>
      </div>
      


      {/* Chart and Recent Sessions Section */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4 bg-white border border-gray-200 shadow-sm">
          <CardHeader className="pb-4">
            <div className="flex justify-between items-start">
              <div>
                <CardTitle className="text-xl font-semibold text-gray-900">Attendance Overview</CardTitle>
                <CardDescription className="text-gray-600 mt-1">
                  {timePeriod === 'daily' ? 'Daily' : timePeriod === 'weekly' ? 'Weekly' : 'Monthly'} attendance trends
                </CardDescription>
              </div>
              <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
                {(['daily', 'weekly', 'monthly'] as const).map((period) => (
                  <button
                    key={period}
                    onClick={() => setTimePeriod(period)}
                    className={`px-4 py-2 text-sm rounded-md transition-colors font-medium ${
                      timePeriod === period
                        ? 'bg-white text-gray-900 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    {period.charAt(0).toUpperCase() + period.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </CardHeader>
          <CardContent className="h-[300px] pl-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartDataWithPercentage}
                margin={{
                  top: 5,
                  right: 20,
                  left: 20,
                  bottom: 5,
                }}
                barGap={0}
                barCategoryGap="20%"
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ 
                    fill: '#6b7280',
                    textAnchor: 'end',
                    dx: -5
                  }}
                  tickFormatter={(value) => `${value}%`}
                  domain={[0, 100]}
                  width={40}
                  tickMargin={0}
                />
                <XAxis 
                  dataKey="name" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ 
                    fill: '#6b7280',
                    dy: 5
                  }}
                  padding={{ left: 15, right: 15 }}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'white',
                    borderRadius: '0.5rem',
                    border: '1px solid #e5e7eb',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    padding: '0.75rem',
                  }}
                  formatter={(value, name) => {
                    const index = chartDataWithPercentage.findIndex(item => 
                      item.presentPercentage === value || item.absentPercentage === value
                    );
                    const data = chartDataWithPercentage[index];
                    const total = data.present + data.absent;
                    const count = name === 'Present' ? data.present : data.absent;
                    return [`${value}% (${count}/${total})`, name];
                  }}
                  labelFormatter={(label) => `Period: ${label}`}
                />
                <Legend 
                  verticalAlign="top"
                  height={36}
                  formatter={(value) => (
                    <span className="text-sm text-muted-foreground">
                      {value} ({value === 'Present' ? '↑' : '↓'})
                    </span>
                  )}
                />
                <Bar 
                  dataKey="presentPercentage" 
                  name="Present"
                  fill="#10b981"
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  dataKey="absentPercentage" 
                  name="Absent"
                  fill="#ef4444"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Recent Attendance Sessions */}
        <Card className="col-span-3 bg-white border border-gray-200 shadow-sm">
          <CardHeader className="pb-3">
            <div className="flex justify-between items-center">
              <div>
                <CardTitle className="text-lg font-semibold text-gray-900">Recent Sessions</CardTitle>
                <CardDescription className="text-gray-600">
                  Latest attendance activities
                </CardDescription>
              </div>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => navigate('/reports')}
                className="h-8 px-3 text-xs"
              >
                View All
              </Button>
            </div>
          </CardHeader>
          <CardContent className="h-[300px] overflow-y-auto">
            <div className="space-y-3">
              {recentSessions.length > 0 ? (
                recentSessions.map((session, index) => (
                  <div 
                    key={index}
                    className="flex items-center p-3 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer border border-transparent hover:border-gray-200"
                    onClick={() => handleSessionClick(session)}
                  >
                    <div className={`p-2 rounded-lg mr-3 ${
                      session.status === 'completed' ? 'bg-green-100' : 
                      session.status === 'ongoing' ? 'bg-blue-100' : 'bg-orange-100'
                    }`}>
                      {session.status === 'completed' ? (
                        <CheckCircle className="h-4 w-4 text-green-700" />
                      ) : session.status === 'ongoing' ? (
                        <CalendarClock className="h-4 w-4 text-blue-700" />
                      ) : (
                        <CalendarDays className="h-4 w-4 text-orange-700" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {session.course}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <p className="text-xs text-gray-500">
                          {session.students} students • {session.time}
                        </p>
                        {session.attendanceRate && (
                          <Badge variant="secondary" className="text-xs px-2 py-0.5">
                            {session.attendanceRate}% attendance
                          </Badge>
                        )}
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 ml-2 text-right">
                      <div>{session.timeAgo}</div>
                      {session.status === 'ongoing' && (
                        <Badge className="bg-blue-100 text-blue-700 text-xs mt-1">
                          Live
                        </Badge>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <div className="p-3 bg-gray-50 rounded-lg inline-block mb-3">
                    <CalendarDays className="h-6 w-6 text-gray-400" />
                  </div>
                  <p className="text-sm text-gray-500">No recent sessions</p>
                  <p className="text-xs text-gray-400 mt-1">Attendance sessions will appear here</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;