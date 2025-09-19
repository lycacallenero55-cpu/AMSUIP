import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { 
  Search, 
  Filter, 
  Download, 
  Eye, 
  Calendar,
  Users,
  BookOpen,
  Star
} from "lucide-react";

const Reports = () => {
  const mockReports = [
    {
      id: 1,
      title: "Computer Science 101",
      type: "class",
      program: "Computer Science",
      date: "2024-01-29",
      timeIn: "09:00",
      timeOut: "10:30",
      totalStudents: 45,
      present: 42,
      absent: 3,
      attendanceRate: 93
    },
    {
      id: 2,
      title: "Annual Science Fair",
      type: "event",
      program: "Science",
      date: "2024-01-28",
      timeIn: "14:00",
      timeOut: "17:00",
      totalStudents: 156,
      present: 142,
      absent: 14,
      attendanceRate: 91
    },
    {
      id: 3,
      title: "Leadership Workshop",
      type: "other",
      program: "Administration",
      date: "2024-01-27",
      timeIn: "10:00",
      timeOut: "12:00",
      totalStudents: 28,
      present: 25,
      absent: 3,
      attendanceRate: 89
    },
    {
      id: 4,
      title: "Mathematics 201",
      type: "class",
      program: "Mathematics",
      date: "2024-01-26",
      timeIn: "11:00",
      timeOut: "12:30",
      totalStudents: 38,
      present: 35,
      absent: 3,
      attendanceRate: 92
    }
  ];

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "class": return <BookOpen className="w-4 h-4" />;
      case "event": return <Calendar className="w-4 h-4" />;
      case "other": return <Star className="w-4 h-4" />;
      default: return <BookOpen className="w-4 h-4" />;
    }
  };

  const getTypeBadge = (type: string) => {
    switch (type) {
      case "class": 
        return <Badge className="bg-primary/10 text-primary border-primary/20">Class</Badge>;
      case "event": 
        return <Badge className="bg-accent/10 text-accent border-accent/20">Event</Badge>;
      case "other": 
        return <Badge className="bg-education-navy/10 text-education-navy border-education-navy/20">Activity</Badge>;
      default: 
        return <Badge>Unknown</Badge>;
    }
  };

  const getAttendanceColor = (rate: number) => {
    if (rate >= 90) return "text-accent";
    if (rate >= 80) return "text-primary";
    return "text-destructive";
  };

  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4 space-y-4">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
          <div>
            <h1 className="text-lg font-bold text-education-navy">ATTENDANCE REPORTS</h1>
            <p className="text-sm text-muted-foreground">
              View and manage all attendance sessions and reports
            </p>
          </div>
          <Button size="sm" className="bg-gradient-primary shadow-glow h-9">
            <Download className="w-3.5 h-3.5 mr-1.5" />
            <span className="text-sm">Export Data</span>
          </Button>
        </div>

        {/* Space between title and first card */}
        <div className="mb-16"></div>

        {/* Filters */}
        <Card className="bg-gradient-card border border-gray-200 shadow-card">
          <CardContent className="p-4">
            <div className="flex items-center justify-between gap-4">
              <div className="flex flex-1 items-center gap-3">
                <div className="relative min-w-[280px] max-w-[400px]">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                  <Input 
                    placeholder="Search reports..." 
                    className="pl-10 h-10 w-full text-sm bg-background border-border focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200"
                  />
                </div>
                <Select>
                  <SelectTrigger className="h-10 min-w-[140px] border-border hover:bg-muted/80 transition-colors duration-200">
                    <SelectValue placeholder="Filter by type" />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border shadow-elegant">
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="class">Classes</SelectItem>
                    <SelectItem value="event">Events</SelectItem>
                    <SelectItem value="other">Activities</SelectItem>
                  </SelectContent>
                </Select>
                <Select>
                  <SelectTrigger className="h-10 min-w-[140px] border-border hover:bg-muted/80 transition-colors duration-200">
                    <SelectValue placeholder="Program" />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border shadow-elegant">
                    <SelectItem value="all">All Programs</SelectItem>
                    <SelectItem value="computer-science">Computer Science</SelectItem>
                    <SelectItem value="mathematics">Mathematics</SelectItem>
                    <SelectItem value="science">Science</SelectItem>
                    <SelectItem value="english">English</SelectItem>
                  </SelectContent>
                </Select>
                <Select>
                  <SelectTrigger className="h-10 min-w-[140px] border-border hover:bg-muted/80 transition-colors duration-200">
                    <SelectValue placeholder="Date range" />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border shadow-elegant">
                    <SelectItem value="today">Today</SelectItem>
                    <SelectItem value="week">This Week</SelectItem>
                    <SelectItem value="month">This Month</SelectItem>
                    <SelectItem value="custom">Custom Range</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Reports List */}
        <div className="space-y-3">
          {mockReports.map((report) => (
            <Card key={report.id} className="bg-gradient-card border border-gray-200 shadow-card">
              <CardContent className="p-4">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-gradient-primary/10">
                      {getTypeIcon(report.type)}
                    </div>
                    <div>
                      <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2">
                        <h3 className="text-sm font-semibold text-education-navy">{report.title}</h3>
                        {getTypeBadge(report.type)}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {report.program} • {report.date} • {report.timeIn} - {report.timeOut}
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-4 sm:gap-6">
                    <div className="text-center">
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Users className="w-3.5 h-3.5" />
                        <span>Students</span>
                      </div>
                      <div className="text-sm font-bold text-education-navy">{report.totalStudents}</div>
                    </div>

                    <div className="text-center">
                      <div className="text-xs text-muted-foreground">Present</div>
                      <div className="text-sm font-bold text-accent">{report.present}</div>
                    </div>

                    <div className="text-center">
                      <div className="text-xs text-muted-foreground">Absent</div>
                      <div className="text-sm font-bold text-destructive">{report.absent}</div>
                    </div>

                    <div className="text-center">
                      <div className="text-xs text-muted-foreground">Rate</div>
                      <div className={`text-sm font-bold ${getAttendanceColor(report.attendanceRate)}`}>
                        {report.attendanceRate}%
                      </div>
                    </div>

                    <div className="flex gap-1">
                      <Button variant="outline" size="sm" className="h-8 w-8 p-0">
                        <Eye className="w-3.5 h-3.5" />
                      </Button>
                      <Button variant="outline" size="sm" className="h-8 w-8 p-0">
                        <Download className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Card className="bg-gradient-card border border-gray-200 shadow-card">
            <CardContent className="p-3 text-center">
              <div className="text-xl font-bold text-education-navy">156</div>
              <div className="text-xs text-muted-foreground">Total Sessions</div>
            </CardContent>
          </Card>
          <Card className="bg-gradient-card border border-gray-200 shadow-card">
            <CardContent className="p-3 text-center">
              <div className="text-xl font-bold text-accent">91.2%</div>
              <div className="text-xs text-muted-foreground">Average Attendance</div>
            </CardContent>
          </Card>
          <Card className="bg-gradient-card border border-gray-200 shadow-card">
            <CardContent className="p-3 text-center">
              <div className="text-xl font-bold text-primary">1,248</div>
              <div className="text-xs text-muted-foreground">Students Tracked</div>
            </CardContent>
          </Card>
        </div>
        </div>
      </PageWrapper>
    </Layout>
  );
};

export default Reports;