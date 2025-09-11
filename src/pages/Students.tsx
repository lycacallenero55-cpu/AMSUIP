import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Search, X, UserPlus, FileDown, Trash2, Edit, Check, Download, Mail, Phone, GraduationCap, Table, LayoutGrid, Users, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Loader2 } from "lucide-react";
import { toast } from "sonner";
import StudentImport from "@/components/StudentImport";
import { useState, useEffect, useCallback, useMemo } from "react";
import { supabase } from "@/lib/supabase";
import { debounce } from "lodash";

interface Student {
  id: number;
  student_id: string;
  firstname: string;
  middlename?: string;
  surname: string;
  program: string;
  year: string;
  section: string;
  email?: string;
  contact_no?: string;
  status?: string;
}

interface PaginationState {
  currentPage: number;
  pageSize: number;
  totalCount: number;
  totalPages: number;
}

interface FilterState {
  program: string;
  year: string;
}

const Students = () => {
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState(false);
  const [totalStudentsCount, setTotalStudentsCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState<FilterState>({
    program: '',
    year: ''
  });
  
  // Pagination state
  const [pagination, setPagination] = useState<PaginationState>({
    currentPage: 1,
    pageSize: 50,
    totalCount: 0,
    totalPages: 0
  });

  const [uniquePrograms, setUniquePrograms] = useState<string[]>([]);
  const [uniqueYears, setUniqueYears] = useState<string[]>([]);

  // Performance tracking
  const [isSearching, setIsSearching] = useState(false);

  // Debounced search function
  const debouncedSearch = useMemo(
    () => debounce((searchTerm: string, program: string, year: string) => {
      fetchStudents(searchTerm, program, year, 1);
      setIsSearching(false);
    }, 300),
    []
  );

  // Define a type for student data
  interface StudentProgramData {
    program: string | null;
    year: string | null;
  }

  // Fetch total students count (unfiltered)
  const fetchTotalStudentsCount = useCallback(async () => {
    try {
      const { count, error } = await supabase
        .from('students')
        .select('*', { count: 'exact', head: true });
      
      if (error) {
        console.error('Error fetching total students count:', error);
        return;
      }
      
      setTotalStudentsCount(count || 0);
    } catch (error) {
      console.error('Error fetching total students count:', error);
    }
  }, []);

  // Fetch filter options with pagination
  const fetchFilterOptions = useCallback(async () => {
    try {
      console.log('Fetching filter options...');
      
      let allStudents: StudentProgramData[] = [];
      let page = 0;
      const pageSize = 1000;
      let hasMore = true;
      
      // Fetch all students with pagination
      while (hasMore) {
        const { data, error, count } = await supabase
          .from('students')
          .select('program, year', { count: 'exact' })
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
      
      // Log raw data for debugging
      console.log('Raw student data count:', allStudents.length);
      console.log('Sample student data:', allStudents.slice(0, 3));
      
      // Process programs
      const programSet = new Set<string>();
      const yearSet = new Set<string>();
      
      allStudents.forEach(student => {
        // Process program
        if (student.program) {
          const program = student.program.toString().trim();
          if (program) programSet.add(program);
        }
        
        // Process year
        if (student.year) {
          const year = student.year.toString().trim();
          if (year) yearSet.add(year);
        }
      });
      
      // Convert sets to sorted arrays
      const programs = Array.from(programSet).sort((a, b) => 
        a.localeCompare(b, 'en', { sensitivity: 'base' })
      );
      
      const years = Array.from(yearSet).sort();
      
      console.log('Unique programs found:', programs);
      console.log('Total unique programs:', programs.length);
      console.log('Unique years found:', years);
      
      setUniquePrograms(programs);
      setUniqueYears(years);
      
    } catch (error) {
      console.error('Error fetching filter options:', error);
      toast.error('Failed to load filter options');
    }
  }, []); // No dependencies - we always want to fetch fresh data

  // Fetch filter options when component mounts
  useEffect(() => {
    fetchFilterOptions();
  }, [fetchFilterOptions]);

  // Main fetch function with server-side pagination and filtering
  const fetchStudents = useCallback(async (
    search = '', 
    program = '', 
    year = '', 
    page = 1,
    pageSize = 50
  ) => {
    try {
      setLoading(true);
      const startTime = performance.now();
      
      // Build Supabase query with count
      let query = supabase
        .from('students')
        .select('*', { count: 'exact' })
        .order('surname', { ascending: true });
      
      // Apply search filter on server side
      if (search && search.trim()) {
        const searchTerm = search.trim();
        query = query.or(`surname.ilike.%${searchTerm}%,firstname.ilike.%${searchTerm}%,student_id.ilike.%${searchTerm}%,middlename.ilike.%${searchTerm}%`);
      }
      
      // Apply program filter
      if (program && program !== 'all') {
        query = query.eq('program', program);
      }
      
      // Apply year filter
      if (year && year !== 'all') {
        query = query.eq('year', year);
      }
      
      // Apply pagination
      const from = (page - 1) * pageSize;
      const to = from + pageSize - 1;
      query = query.range(from, to);
      
      const { data, error, count } = await query;
      
      if (error) {
        console.error('Supabase error:', error);
        throw new Error(error.message || 'Failed to fetch students');
      }
      
      // Transform the data to match our Student interface
      const formattedStudents: Student[] = Array.isArray(data) 
        ? data.map((student) => ({
            ...student,
            middlename: student.middlename || '',
            email: student.email || '',
            contact_no: student.contact_no || ''
          }))
        : [];
      
      const totalCount = count || 0;
      const totalPages = Math.ceil(totalCount / pageSize);
      
      setStudents(formattedStudents);
      setPagination({
        currentPage: page,
        pageSize,
        totalCount,
        totalPages
      });
      
      const endTime = performance.now();
      console.log(`Query completed in ${(endTime - startTime).toFixed(2)}ms`);
      
    } catch (error) {
      console.error('Error fetching students:', error);
      toast.error('Failed to load students. Please try again.');
      setStudents([]);
      setPagination(prev => ({ ...prev, totalCount: 0, totalPages: 0 }));
    } finally {
      setLoading(false);
    }
  }, []);

  // Handle search input change
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchTerm(value);
    setIsSearching(true);
    
    // Reset to first page when searching
    setPagination(prev => ({ ...prev, currentPage: 1 }));
    
    // Debounced search
    debouncedSearch(value, filters.program, filters.year);
  };

  // Handle filter changes
  const handleFilterChange = (filterName: keyof FilterState, value: string) => {
    const newFilters = {
      ...filters,
      [filterName]: value === 'all' ? '' : value
    };
    
    setFilters(newFilters);
    
    // Reset to first page when filtering
    setPagination(prev => ({ ...prev, currentPage: 1 }));
    
    // Apply filters immediately
    fetchStudents(searchTerm, newFilters.program, newFilters.year, 1, pagination.pageSize);
  };

  // Handle page change
  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > pagination.totalPages) return;
    
    setPagination(prev => ({ ...prev, currentPage: newPage }));
    fetchStudents(searchTerm, filters.program, filters.year, newPage, pagination.pageSize);
  };

  // Handle page size change
  const handlePageSizeChange = (newPageSize: number) => {
    setPagination(prev => ({ 
      ...prev, 
      pageSize: newPageSize, 
      currentPage: 1 
    }));
    fetchStudents(searchTerm, filters.program, filters.year, 1, newPageSize);
  };

  // Clear all filters
  const clearFilters = () => {
    setSearchTerm('');
    setFilters({ program: '', year: '' });
    setPagination(prev => ({ ...prev, currentPage: 1 }));
    setIsSearching(false);
    fetchStudents('', '', '', 1, pagination.pageSize);
  };

  // Initialize component
  useEffect(() => {
    fetchFilterOptions();
    fetchStudents('', '', '', 1, 50);
    fetchTotalStudentsCount();
  }, []);

  // Cleanup debounced function
  useEffect(() => {
    return () => {
      debouncedSearch.cancel();
    };
  }, [debouncedSearch]);

  const handleStudentAdded = () => {
    // Refresh current page
    fetchStudents(searchTerm, filters.program, filters.year, pagination.currentPage, pagination.pageSize);
    // Refresh total count
    fetchTotalStudentsCount();
    toast.success('Student added successfully!');
  };

  const handleStudentDeleted = (id: number) => {
    // Optimistically update UI
    setStudents(prevStudents => prevStudents.filter(student => student.id !== id));
    setPagination(prev => ({ 
      ...prev, 
      totalCount: Math.max(0, prev.totalCount - 1),
      totalPages: Math.ceil(Math.max(0, prev.totalCount - 1) / prev.pageSize)
    }));
    // Update total students count
    setTotalStudentsCount(prev => Math.max(0, prev - 1));
    toast.success('Student deleted successfully!');
    
    // Refresh to ensure consistency
    setTimeout(() => {
      fetchStudents(searchTerm, filters.program, filters.year, pagination.currentPage, pagination.pageSize);
      fetchTotalStudentsCount();
    }, 500);
  };


  const getInitials = (name: string) => {
    if (!name) return '';
    return name
      .split(' ')
      .filter(part => part.length > 0)
      .map(part => part[0])
      .join('')
      .toUpperCase();
  };

  const getAttendanceColor = (rate: number) => {
    if (rate >= 90) return "text-accent";
    if (rate >= 80) return "text-primary";
    return "text-destructive";
  };

  const getStatusBadge = (status: string) => {
    return status === "Active" 
      ? <Badge className="bg-accent/10 text-accent border-accent/20">Active</Badge>
      : <Badge className="bg-muted text-muted-foreground">Inactive</Badge>;
  };

  // Pagination component
  const PaginationControls = () => {
    const { currentPage, totalPages, totalCount, pageSize } = pagination;
    
    if (totalPages <= 1) return null;

    const getVisiblePages = () => {
      const delta = 2;
      const range = [];
      const rangeWithDots = [];

      for (let i = Math.max(2, currentPage - delta); i <= Math.min(totalPages - 1, currentPage + delta); i++) {
        range.push(i);
      }

      if (currentPage - delta > 2) {
        rangeWithDots.push(1, '...');
      } else {
        rangeWithDots.push(1);
      }

      rangeWithDots.push(...range);

      if (currentPage + delta < totalPages - 1) {
        rangeWithDots.push('...', totalPages);
      } else if (totalPages > 1) {
        rangeWithDots.push(totalPages);
      }

      return rangeWithDots;
    };

    const startItem = ((currentPage - 1) * pageSize) + 1;
    const endItem = Math.min(currentPage * pageSize, totalCount);

    return (
      <div className="flex items-center justify-between px-4 py-3 bg-white border-t border-gray-200">
        <div className="flex items-center text-sm text-gray-700">
          <span>Showing {startItem.toLocaleString()} to {endItem.toLocaleString()} of {totalCount.toLocaleString()} students</span>
          <div className="ml-4 flex items-center space-x-2">
            <span>Show:</span>
            <Select
              value={pageSize.toString()}
              onValueChange={(value) => handlePageSizeChange(parseInt(value))}
            >
              <SelectTrigger className="w-20 h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="25">25</SelectItem>
                <SelectItem value="50">50</SelectItem>
                <SelectItem value="100">100</SelectItem>
                <SelectItem value="200">200</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex items-center space-x-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(1)}
            disabled={currentPage === 1}
            className="h-8 w-8 p-0"
          >
            <ChevronsLeft className="h-4 w-4" />
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(currentPage - 1)}
            disabled={currentPage === 1}
            className="h-8 w-8 p-0"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          {getVisiblePages().map((page, index) => (
            <Button
              key={index}
              variant={currentPage === page ? "default" : "outline"}
              size="sm"
              onClick={() => typeof page === 'number' && handlePageChange(page)}
              disabled={typeof page === 'string'}
              className="h-8 min-w-[32px] px-2"
            >
              {page}
            </Button>
          ))}

          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="h-8 w-8 p-0"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => handlePageChange(totalPages)}
            disabled={currentPage === totalPages}
            className="h-8 w-8 p-0"
          >
            <ChevronsRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  };

  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4">
        <div className="mb-3">
          <div className="flex flex-col">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-1">
              <div>
                <h1 className="text-lg font-bold text-education-navy">STUDENTS</h1>
                <p className="text-sm text-muted-foreground">
                  Manage and monitor student records
                </p>
              </div>
              <div className="mt-2 md:mt-0">
                <StudentImport 
                  onImportComplete={handleStudentAdded}
                  onImportSuccess={handleStudentAdded}
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Search and Students Section */}
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="mb-3">
            <h3 className="text-base font-semibold text-education-navy">List of Students {pagination.currentPage > 1 && `(Page ${pagination.currentPage})`}</h3>
          </div>
          <div className="flex items-center justify-between gap-4 p-0 mb-4">
            <div className="flex flex-1 items-center gap-3">
              <div className="relative min-w-[280px] max-w-[400px]">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                {isSearching && (
                  <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-primary animate-spin" />
                )}
                <Input
                  placeholder="Search by name or ID..."
                  className="pl-10 pr-10 h-10 w-full text-sm bg-background border-border focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200 [&::-webkit-search-cancel-button]:hidden"
                  value={searchTerm}
                  onChange={handleSearchChange}
                  type="search"
                />
              </div>
            </div>
          </div>
          
          <div className="text-sm text-gray-500 flex items-center justify-between mb-3">
            <span>
              {loading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Loading students...
                </span>
              ) : (
                <>
                  Showing {students.length} of {pagination.totalCount.toLocaleString()} students
                  {(searchTerm || filters.program || filters.year) && (
                    <span>
                      {' '}matching current filters
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="ml-2 h-auto p-0 text-blue-600 hover:bg-accent/30 hover:text-blue-700 hover:underline transition-colors duration-200 rounded-sm px-1.5 py-0.5"
                        onClick={clearFilters}
                      >
                        Clear all
                      </Button>
                    </span>
                  )}
                </>
              )}
            </span>
          </div>
          
        
          {/* Table View */}
          <div className="border-t border-b border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr className="text-xs text-gray-500">
                  <th scope="col" className="px-3 py-2 text-left font-medium">Student</th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">ID</th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Program</th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Year & Section</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200 text-sm">
                {loading ? (
                <tr>
                  <td colSpan={4} className="px-3 py-6 text-center">
                    <div className="flex justify-center">
                      <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
                    </div>
                    <p className="mt-2 text-sm text-gray-500">Loading students...</p>
                  </td>
                </tr>
                ) : students.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-3 py-8 text-center text-sm text-gray-500">
                    {pagination.totalCount === 0 
                      ? 'No students found. Add your first student!'
                      : 'No students match the current filters. Try adjusting your search or filters.'}
                    {pagination.totalCount > 0 && (
                      <Button 
                        variant="outline" 
                        size="sm"
                        className="mt-3 h-8 text-xs hover:bg-gray-100 hover:text-gray-900"
                        onClick={clearFilters}
                      >
                        Clear all filters
                      </Button>
                    )}
                  </td>
                </tr>
                ) : (
                  students.map((student) => (
                    <tr key={student.id} className="hover:bg-gray-50">
                      <td className="px-3 py-2 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {student.surname}, {student.firstname}{student.middlename ? ' ' + student.middlename.charAt(0) + '.' : ''}
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap text-gray-500 text-sm">
                        {student.student_id}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap text-gray-500 text-sm">
                        <span className="truncate max-w-[120px] inline-block">{student.program}</span>
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap text-gray-500 text-sm">
                        <div className="flex items-center gap-1">
                          <span>{student.year}</span>
                          <span className="text-gray-300">•</span>
                          <span> {student.section || 'N/A'}</span>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          <PaginationControls />
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8">
        <Card className="bg-gradient-card border-0 shadow-card">
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-gray-900">{totalStudentsCount.toLocaleString()}</div>
            <div className="text-sm text-gray-500">Total Students</div>
          </CardContent>
        </Card>
        <Card className="bg-gradient-card border-0 shadow-card">
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-accent">
              {students.filter(student => student.status === 'Active').length.toLocaleString()}
            </div>
            <div className="text-sm text-muted-foreground">Active Students (Current Page)</div>
          </CardContent>
        </Card>
        <Card className="bg-gradient-card border-0 shadow-card">
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-primary">89.2%</div>
            <div className="text-sm text-muted-foreground">Avg Attendance</div>
          </CardContent>
        </Card>
        <Card className="bg-gradient-card border-0 shadow-card">
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-education-green">
              {uniquePrograms.length}
            </div>
            <div className="text-sm text-muted-foreground">Programs</div>
          </CardContent>
        </Card>
      </div>
      </PageWrapper>
    </Layout>
  );
}

export default Students;