import { useState, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { CalendarIcon, Clock, Users, BookOpen, Calendar, Star, Loader2, CalendarClock } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { fetchStudents } from "@/lib/supabaseService";
import { useAuth } from "@/hooks/useAuth";
import { fetchUserRole } from "@/lib/getUserRole";

export type AttendanceType = "class" | "event" | "other";

export interface SessionData {
  title: string;
  program: string;
  year: string;
  section: string;
  date: string;
  timeIn: string;
  timeOut: string;
  attendanceType: AttendanceType;
  venue?: string;
  description?: string;
  capacity?: string | number;
}

interface AttendanceFormProps {
  onSuccess?: () => void;
  onSubmit?: (session: SessionData) => void;
  initialData?: Partial<SessionData> & { id?: number };
}

const AttendanceForm = ({ onSuccess, onSubmit, initialData }: AttendanceFormProps) => {
  const { user } = useAuth();

  const [currentRole, setCurrentRole] = useState<string>("");
  const [roleReady, setRoleReady] = useState<boolean>(false);

  // Read cached role (same key used elsewhere)
  const getCachedUserRole = () => {
    try {
      return localStorage.getItem('userRole') || '';
    } catch {
      return '';
    }
  };

  useEffect(() => {
    let isMounted = true;
    const resolveRole = async () => {
      // 1) Set cached role immediately for instant UI
      const cached = getCachedUserRole();
      if (isMounted && cached) {
        setCurrentRole(cached);
        setRoleReady(true);
      }

      // 2) Get authoritative role from database
      if (user?.id) {
        try {
          const dbRole = await fetchUserRole(user.id);
          if (isMounted && dbRole) {
            // Only update if the role actually changed
            setCurrentRole(prevRole => {
              if (prevRole !== dbRole) {
                return dbRole;
              }
              return prevRole;
            });
          }
        } catch (error) {
          console.error('Error resolving user role:', error);
        }
      }

      // 3) Ensure roleReady is set even if no cached role
      if (isMounted && !cached) {
        setRoleReady(true);
      }
    };
    
    resolveRole();
    return () => {
      isMounted = false;
    };
  }, [user?.id]);

  const allowedTypes: AttendanceType[] = useMemo(() => {
    // Use the exact normalized role values from getUserRole.ts
    if (currentRole === "admin" || currentRole === "ROTC admin") {
      return ["class", "event", "other"];
    }
    
    if (currentRole === "Instructor") {
      return ["class"];
    }
    
    if (currentRole === "SSG officer" || currentRole === "ROTC officer") {
      return ["event", "other"];
    }
    
    // Default: Show reasonable initial types based on cached role to prevent flicker
    if (!currentRole && !roleReady) {
      const cached = getCachedUserRole();
      if (cached === "admin" || cached === "ROTC admin") {
        return ["class", "event", "other"];
      }
      if (cached === "Instructor") {
        return ["class"];
      }
      if (cached === "SSG officer" || cached === "ROTC officer") {
        return ["event", "other"];
      }
    }
    
    // Fallback for truly unknown roles
    return ["class"];
  }, [currentRole, roleReady]);

  const [attendanceType, setAttendanceType] = useState<AttendanceType>(() => {
    if (initialData?.attendanceType) {
      return initialData.attendanceType;
    }
    // Set default based on cached role if available
    const cached = getCachedUserRole();
    if (cached === "SSG officer" || cached === "ROTC officer") {
      return "event";
    }
    return "class";
  });
  const [formData, setFormData] = useState({
    title: initialData?.title || "",
    program: initialData?.program || "",
    year: initialData?.year || "",
    section: initialData?.section || "",
    date: initialData?.date || "",
    timeIn: initialData?.timeIn || "",
    timeOut: initialData?.timeOut || "",
  });
  
  // Ensure selected type is allowed for the current role, but only change if truly necessary
  useEffect(() => {
    if (roleReady && allowedTypes.length > 0) {
      // Only change if the current selection is not allowed
      if (!allowedTypes.includes(attendanceType)) {
        // For SSG officers, prefer "event" as default, for instructors prefer "class"
        if (currentRole === "SSG officer" || currentRole === "ROTC officer") {
          setAttendanceType(allowedTypes.includes("event") ? "event" : allowedTypes[0]);
        } else {
          setAttendanceType(allowedTypes[0]);
        }
      }
    }
  }, [roleReady, allowedTypes, attendanceType, currentRole]);
  
  // State for dropdown options
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [studentOptions, setStudentOptions] = useState<{
    programs: string[];
    years: string[];
    sections: { [key: string]: string[] };
  }>({ programs: [], years: [], sections: {} });
  
  // Available sections based on selected program and year
  const availableSections = useCallback(() => {
    if (!formData.program || !formData.year) return [];
    const key = `${formData.program}|${formData.year}`;
    return studentOptions.sections[key] || [];
  }, [formData.program, formData.year, studentOptions.sections]);
  
  // Fetch students for a specific program and year
  const fetchStudents = async (program: string, year: string) => {
    try {
      // Convert year to the format stored in the database (e.g., '1st' instead of '1st Year')
      const yearShort = year.replace(' Year', '');
      
      // Helper function to fetch all students with pagination
      const fetchAllStudents = async () => {
        interface StudentSection {
          section: string | null;
        }
        
        const allStudents: StudentSection[] = [];
        let from = 0;
        const pageSize = 1000; // Supabase default limit
        
        while (true) {
          const { data, error } = await supabase
            .from('students')
            .select('section')
            .eq('program', program)
            .eq('year', yearShort)
            .not('section', 'is', null)
            .range(from, from + pageSize - 1);
          
          if (error) throw error;
          
          if (!data || data.length === 0) break;
          
          allStudents.push(...data);
          
          // If we got less than pageSize, we've reached the end
          if (data.length < pageSize) break;
          
          from += pageSize;
        }
        
        return allStudents;
      };
      
      // Fetch all students with pagination
      const data = await fetchAllStudents();
      
      // Return an array of sections with empty strings converted to 'Uncategorized'
      return (data || []).map(student => ({
        section: student.section || 'Uncategorized'
      }));
    } catch (error) {
      console.error('Error fetching students:', error);
      toast.error('Failed to load student data');
      return [];
    }
  };

  // Load student sections when program or year changes - debounced to prevent blinking
  const loadStudentSections = useCallback(async (program: string, year: string) => {
    // Skip if no program or year is selected, or if "All Programs" or "All Year Levels" is selected
    if (!program || !year || program === 'All Programs' || year === 'All Year Levels') {
      const key = `${program}|${year}`;
      setStudentOptions(prev => {
        // Only update if sections actually change
        const currentSections = prev.sections[key] || [];
        if (currentSections.length === 0) {
          return prev;
        }
        return {
          ...prev,
          sections: {
            ...prev.sections,
            [key]: []
          }
        };
      });
      return;
    }
    
    const key = `${program}|${year}`;
    
    // Check if we already have sections for this combination
    if (studentOptions.sections[key] && studentOptions.sections[key].length > 0) {
      return;
    }
    
    setLoadingOptions(true);
    try {
      // Fetch students for the selected program and year
      const students = await fetchStudents(program, year);
      
      // Extract unique sections from the students and filter out any null/undefined values
      const sections = [...new Set(students.map(student => student.section).filter(Boolean))];
      
      // If no sections found, add a default section
      if (sections.length === 0) {
        sections.push('Default Section');
      }
      
      setStudentOptions(prev => ({
        ...prev,
        sections: {
          ...prev.sections,
          [key]: sections
        }
      }));
    } catch (error) {
      console.error('Error in loadStudentSections:', error);
      toast.error('Failed to load student sections');
    } finally {
      setLoadingOptions(false);
    }
  }, [studentOptions.sections]);

  // Fetch programs from students table (runs once on component mount)
  const loadPrograms = useCallback(async () => {
    console.log('Starting to load programs...');
    
    try {
      console.log('Fetching programs from students table...');
      
      let allStudents: { program: string | null }[] = [];
      let page = 0;
      const pageSize = 1000;
      let hasMore = true;
      
      // Fetch all students with pagination
      while (hasMore) {
        const { data, error } = await supabase
          .from('students')
          .select('program')
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
      
      let programs = Array.from(programSet).sort((a, b) => 
        a.localeCompare(b, 'en', { sensitivity: 'base' })
      );
      
      console.log('Unique programs found in students table:', programs);
      
      // Ensure we have programs
      if (programs.length === 0) {
        console.warn('No programs found in database');
        programs = [];
      }
      
      console.log('Final programs to set:', programs);
      
      // Update state only if programs have changed
      setStudentOptions(prev => {
        if (JSON.stringify(prev.programs) === JSON.stringify(programs)) {
          console.log('Programs unchanged, skipping state update');
          return prev;
        }
        
        return {
          ...prev,
          programs
        };
      });
    } catch (error) {
      console.error('Error in loadPrograms:', error);
      toast.error('Failed to load programs. Check console for details.');
    }
  }, []); // Remove all dependencies to prevent re-creation

  // Fetch years - include up to 4th year with consistent formatting
  const loadYears = useCallback(async () => {
    try {
      console.log('Loading years...');
      
      // Define standard year levels up to 4th year
      const standardYears = ['1st Year', '2nd Year', '3rd Year', '4th Year'];
      
      // Also fetch any additional years that might exist in the database
      const { data, error } = await supabase
        .from('students')
        .select('year')
        .not('year', 'is', null)
        .order('year');
      
      if (error) {
        console.error('Error fetching years from database:', error);
        // Fall back to standard years if there's an error
        setStudentOptions(prev => ({
          ...prev,
          years: [...standardYears]
        }));
        return;
      }
      
      // Get unique years from database and normalize their format
      const dbYears = [...new Set(data.map(item => {
        const year = item.year?.toString().trim();
        // Convert short forms to full forms (e.g., '1st' -> '1st Year')
        if (year?.match(/^\d+(st|nd|rd|th)$/i)) {
          return `${year} Year`;
        }
        return year;
      }))].filter(Boolean) as string[];
      
      // Combine with standard years, remove duplicates, and sort
      const allYears = Array.from(new Set([...standardYears, ...dbYears]))
        .filter(Boolean)
        .sort((a, b) => {
          // Sort by year number
          const getYearNum = (str: string) => parseInt(str, 10);
          return getYearNum(a) - getYearNum(b);
        });
      
      console.log('Available years:', allYears);
      
      setStudentOptions(prev => ({
        ...prev,
        years: allYears
      }));
    } catch (error) {
      console.error('Error fetching years:', error);
      toast.error("Failed to load year levels. Please try again later.");
    }
  }, []);

  // Load programs and years on component mount - prevent multiple calls
  useEffect(() => {
    console.log('Component mounted, initializing data...');
    let isMounted = true;
    
    const initializeData = async () => {
      try {
        console.log('Starting to load programs and years...');
        // Load programs and years in parallel but only once
        await Promise.all([loadPrograms(), loadYears()]);
        if (isMounted) {
          console.log('Programs and years loaded successfully');
        }
      } catch (error) {
        console.error('Error initializing form data:', error);
      }
    };
    
    // Only initialize if we don't have data yet
    if (studentOptions.programs.length === 0 && studentOptions.years.length === 0) {
      initializeData();
    }
    
    return () => {
      isMounted = false;
      console.log('Component unmounted, cleaning up...');
    };
  }, []); // Remove dependencies to prevent re-running
  
  // Load sections when program or year changes - but only when both are available and different
  useEffect(() => {
    if (formData.program && formData.year && 
        formData.program !== 'All Programs' && 
        formData.year !== 'All Year Levels') {
      const key = `${formData.program}|${formData.year}`;
      // Only load if we don't already have sections for this combination
      if (!studentOptions.sections[key] || studentOptions.sections[key].length === 0) {
        loadStudentSections(formData.program, formData.year);
      }
    }
  }, [formData.program, formData.year]); // Simplified dependencies

  // Reset year and section when program changes - optimized to prevent unnecessary re-renders
  useEffect(() => {
    if (formData.program && formData.year) {
      // If we have both program and year, ensure the section is valid
      const sections = availableSections();
      if (formData.section && sections.length > 0 && !sections.includes(formData.section)) {
        setFormData(prev => ({
          ...prev,
          section: ""
        }));
      }
    }
  }, [formData.program, formData.year, studentOptions.sections]);

  // Handle initial data changes separately to prevent conflicts with form state
  useEffect(() => {
    if (initialData) {
      const newFormData = {
        title: initialData.title || "",
        program: initialData.program || "",
        year: initialData.year || "",
        section: initialData.section || "",
        date: initialData.date || "",
        timeIn: initialData.timeIn || "",
        timeOut: initialData.timeOut || "",
      };
      
      // Only update if data has actually changed
      const hasChanged = Object.keys(newFormData).some(key => 
        formData[key as keyof typeof formData] !== newFormData[key as keyof typeof newFormData]
      );
      
      if (hasChanged) {
        setFormData(newFormData);
      }
      
      if (initialData.attendanceType && initialData.attendanceType !== attendanceType) {
        setAttendanceType(initialData.attendanceType);
      }
    }
  }, [initialData?.id]); // Only depend on the ID change, not the entire initialData object

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Prepare the session data
    const sessionData: SessionData = {
      ...formData,
      // Map 'All Programs' and 'All Year Levels' to empty strings for backend processing
      program: formData.program === 'All Programs' ? 'All Programs' : formData.program,
      year: formData.year === 'All Year Levels' ? 'All Years' : formData.year,
      section: formData.section === 'All Sections' ? '' : formData.section,
      attendanceType: attendanceType
    };
    
    try {
      // Call the onSubmit callback if provided (for parent component handling)
      if (onSubmit) {
        onSubmit(sessionData);
      }
      
      // Show success message
      toast.success(`${formData.title} session has been ${initialData?.id ? 'updated' : 'created'} successfully.`);
      
      // Only reset form if we're not in edit mode
      if (!initialData?.id) {
        setFormData({
          title: "",
          program: "",
          year: "",
          section: "",
          date: "",
          timeIn: "",
          timeOut: "",
        });
        setAttendanceType("class");
      }
      
      // Call onSuccess callback if provided
      if (onSuccess) {
        onSuccess();
      }
    } catch (error) {
      console.error("Error saving session:", error);
      toast.error(`Failed to ${initialData?.id ? 'update' : 'create'} session. Please try again.`);
    }
  };

  const getTypeIcon = (type: AttendanceType) => {
    switch (type) {
      case "class": return <BookOpen className="w-5 h-5" />;
      case "event": return <Calendar className="w-5 h-5" />;
      case "other": return <Star className="w-5 h-5" />;
    }
  };

  const getTypeColor = (type: AttendanceType) => {
    switch (type) {
      case "class": return "bg-gradient-primary";
      case "event": return "bg-gradient-accent";
      case "other": return "bg-education-navy";
    }
  };

  return (
    <div className="w-full max-w-none mx-auto px-6 py-4">
      <div className="border-b border-border/30 pb-2 mb-3 flex justify-between items-center">
        <h2 className="flex items-center gap-2 text-education-navy text-xl font-semibold">
          <CalendarClock className="w-5 h-5 text-education-blue" />
          {initialData?.id ? 'Edit Session' : 'Create New Session'}
        </h2>
      </div>
      
      {/* Type Selection */}
      <div className="pb-3">
        <div
          className="grid gap-4 justify-center mx-auto"
          style={{ gridTemplateColumns: `repeat(${allowedTypes.length}, minmax(12rem, 1fr))`, maxWidth: '900px' }}
        >
            {[
              { type: "class" as AttendanceType, label: "Class Session", description: "Regular classroom attendance", bg: "bg-gradient-primary/5", hoverBg: "hover:bg-gradient-primary/10" },
              { type: "event" as AttendanceType, label: "School Event", description: "Assemblies, ceremonies, programs", bg: "bg-gradient-accent/5", hoverBg: "hover:bg-gradient-accent/10" },
              { type: "other" as AttendanceType, label: "Other Activity", description: "Workshops, field trips, meetings", bg: "bg-education-navy/5", hoverBg: "hover:bg-education-navy/10" }
            ]
              .filter(option => allowedTypes.includes(option.type))
              .map((option) => (
              <div 
                key={option.type} 
                className={`group relative rounded-lg p-0.5 ${option.bg} ${option.hoverBg} transition-all duration-200 hover:scale-105 w-full`}
              >
                <div 
                  className={`p-4 rounded-lg cursor-pointer transition-all duration-200 h-full ${
                    attendanceType === option.type 
                      ? getTypeColor(option.type) 
                      : 'bg-white group-hover:bg-white/90 group-hover:shadow-sm'
                  }`}
                  onClick={() => setAttendanceType(option.type)}
                >
                  <div className="flex flex-col items-center gap-1.5">
                    <div className={`transition-all duration-200 transform ${
                      attendanceType === option.type 
                        ? 'text-white scale-110' 
                        : 'text-muted-foreground group-hover:text-primary group-hover:scale-110'
                    }`}>
                      {getTypeIcon(option.type)}
                    </div>
                    <div className="text-center">
                      <div className={`font-medium text-sm transition-colors duration-200 ${
                        attendanceType === option.type 
                          ? 'text-white' 
                          : 'text-foreground group-hover:text-primary'
                      }`}>
                        {option.label}
                      </div>
                      <div className={`text-xs transition-colors duration-200 ${
                        attendanceType === option.type 
                          ? 'text-white/90' 
                          : 'text-muted-foreground group-hover:text-primary/80'
                      }`}>
                        {option.description}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Dynamic Form */}
      <div className="pb-8">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
            {/* Title */}
            <div className="space-y-2 col-span-2">
              <Label htmlFor="title">
                {attendanceType === "class" ? "Subject/Course Name" : 
                 attendanceType === "event" ? "Event Name" : "Activity Title"}
              </Label>
                <Input
                  id="title"
                  value={formData.title}
                  onChange={(e) => setFormData({...formData, title: e.target.value})}
                  placeholder={attendanceType === "class" ? "e.g., Mathematics 101" : 
                             attendanceType === "event" ? "e.g., Annual Science Fair" : "e.g., Leadership Workshop"}
                  className="w-full focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-primary"
                  autoFocus={false}
                  onFocus={(e) => {
                    // Prevent auto-highlighting of text on focus
                    e.target.setSelectionRange(e.target.value.length, e.target.value.length);
                  }}
                  required
                />
            </div>

            {/* Program */}
            <div className="space-y-2">
              <Label htmlFor="program">Program</Label>
                <Select 
                  value={formData.program}
                  onValueChange={(value) => {
                    setFormData(prev => ({
                      ...prev,
                      program: value,
                      // Only reset year and section if changing to a different program
                      year: prev.program !== value ? "" : prev.year,
                      section: prev.program !== value ? "" : prev.section
                    }));
                  }}
                  disabled={loadingOptions}
                >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder={loadingOptions ? "Loading..." : "Select program"} />
                </SelectTrigger>
                <SelectContent>
                  {loadingOptions ? (
                    <div className="flex items-center justify-center p-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </div>
                  ) : (
                    <>
                      <SelectItem value="All Programs">All Programs</SelectItem>
                      {studentOptions.programs.map((program) => (
                        <SelectItem key={program} value={program}>
                          {program}
                        </SelectItem>
                      ))}
                    </>
                  )}
                </SelectContent>
              </Select>

            </div>

            {/* Year Level - Show for class, event, and other types */}
            {(attendanceType === "class" || attendanceType === "event" || attendanceType === "other") && (
              <div className="space-y-2">
                <Label htmlFor="year">Year Level</Label>
                <Select 
                  value={formData.year === 'All Years' ? 'All Year Levels' : formData.year}
                  onValueChange={(value) => {
                    const newYear = value === 'All Year Levels' ? 'All Years' : value;
                    setFormData(prev => ({
                      ...prev,
                      year: newYear,
                      // Only reset section if year actually changed
                      section: prev.year !== newYear ? "" : prev.section
                    }));
                  }}
                  disabled={loadingOptions}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select year level" />
                  </SelectTrigger>
                  <SelectContent>
                    {loadingOptions ? (
                      <div className="flex items-center justify-center p-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                      </div>
                    ) : (
                      <>
                        <SelectItem value="All Year Levels">All Year Levels</SelectItem>
                        {studentOptions.years.map((year) => (
                          <SelectItem key={year} value={year}>
                            {year}
                          </SelectItem>
                        ))}
                      </>
                    )}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Section - Show for class and other types, but not for events */}
            {(attendanceType === "class" || attendanceType === "other") && (
              <div className="space-y-2">
                <Label htmlFor="section">Section</Label>
                <Select
                  value={formData.section}
                  onValueChange={(value) => setFormData({...formData, section: value})}
                  disabled={
                    !formData.program || 
                    !formData.year ||
                    formData.program === 'All Programs' ||
                    formData.year === 'All Year Levels' ||
                    loadingOptions
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder={
                      availableSections().length === 0 ? "No sections available" : "Select section"
                    } />
                  </SelectTrigger>
                  <SelectContent>
                    {loadingOptions ? (
                      <div className="flex items-center justify-center p-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                      </div>
                    ) : availableSections().length === 0 ? (
                      <div className="p-2 text-sm text-muted-foreground">
                        No sections available for this program/year
                      </div>
                    ) : (
                      availableSections().map((section) => (
                        <SelectItem key={section} value={section}>
                          {section}
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
              </div>
            )}

              {/* Date */}
              <div className="space-y-2">
                <Label htmlFor="date">Date</Label>
                <Input
                  id="date"
                  type="date"
                  value={formData.date}
                  onChange={(e) => setFormData({...formData, date: e.target.value})}
                  className="w-full cursor-pointer"
                  required
                />
              </div>

              {/* Time In */}
              <div className="space-y-2">
                <Label htmlFor="timeIn">Start Time</Label>
                <Input
                  id="timeIn"
                  type="time"
                  value={formData.timeIn}
                  onChange={(e) => setFormData({...formData, timeIn: e.target.value})}
                  className="w-full cursor-pointer"
                  required
                />
              </div>

              {/* Time Out */}
              <div className="space-y-2">
                <Label htmlFor="timeOut">End Time</Label>
                <Input
                  id="timeOut"
                  type="time"
                  value={formData.timeOut}
                  onChange={(e) => setFormData({...formData, timeOut: e.target.value})}
                  className="w-full cursor-pointer"
                />
              </div>

            </div>

            {/* Submit Button */}
            <div className="md:col-span-2 pt-3 pb-4 flex justify-center">
              <Button 
                type="submit" 
                className={`${getTypeColor(attendanceType)} shadow-glow hover:opacity-90 transition-opacity w-full max-w-[400px] min-h-[42px] text-base`}
              >
                <Users className="w-4 h-4 mr-2" />
                {initialData?.id ? 'Update Session' : 'Create Session'}
              </Button>
            </div>
          </form>
        </div>
      </div>
   
  );
};

export default AttendanceForm;