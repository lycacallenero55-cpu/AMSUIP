import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, Loader2, ChevronDown } from "lucide-react";
import type React from "react";
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
  const [isCustomInput, setIsCustomInput] = useState(false);
  
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
    // Ensure minimum value of 10
    const validPageSize = Math.max(10, newPageSize);
    setPagination(prev => ({ 
      ...prev, 
      pageSize: validPageSize, 
      currentPage: 1 
    }));
    fetchStudents(searchTerm, filters.program, filters.year, 1, validPageSize);
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

  // Removed status/attendance helpers from table layout

  // Pagination controls are removed from bottom; page size control moved to top as "Showed".

  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4">
        <div className="mb-3">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-1">
            <div>
              <h1 className="text-2xl font-bold text-education-navy">STUDENTS</h1>
              <p className="text-sm text-muted-foreground">
                Manage and monitor student records
              </p>
            </div>
          </div>
        </div>
        
        {/* Search and Students Section */}
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-base font-semibold text-education-navy">List of Students {pagination.currentPage > 1 && `(Page ${pagination.currentPage})`}</h3>
            <div className="mt-0">
              <StudentImport 
                onImportComplete={handleStudentAdded}
                onImportSuccess={handleStudentAdded}
              />
            </div>
          </div>
          {/* Top controls row */}
          <div className="flex items-center justify-between gap-4 p-0 mb-3">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Showed:</span>
              <div className="relative">
                {isCustomInput ? (
                  <Input
                    type="number"
                    min="10"
                    value={pagination.pageSize}
                    onChange={(e) => {
                      const value = parseInt(e.target.value);
                      if (!isNaN(value) && value >= 10) {
                        handlePageSizeChange(value);
                      }
                    }}
                    onBlur={(e) => {
                      const value = parseInt(e.target.value);
                      if (isNaN(value) || value < 10) {
                        e.target.value = pagination.pageSize.toString();
                      }
                      setIsCustomInput(false);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        setIsCustomInput(false);
                        e.currentTarget.blur();
                      }
                      if (e.key === 'Escape') {
                        setIsCustomInput(false);
                        e.currentTarget.blur();
                      }
                    }}
                    className="h-8 w-24 text-center pr-8"
                    placeholder="50"
                    autoFocus
                  />
                ) : (
                  <div
                    className="h-8 w-24 flex items-center justify-between px-3 border border-input bg-background rounded-md cursor-pointer hover:bg-accent/50 transition-colors"
                    onClick={() => setIsCustomInput(true)}
                  >
                    <span className="text-sm">{pagination.pageSize}</span>
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Search:</span>
              <div className="relative min-w-[240px] max-w-[340px]">
                <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                {isSearching && (
                  <Loader2 className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-primary animate-spin" />
                )}
                <Input
                  placeholder="name or ID..."
                  className="pl-7 pr-7 h-8 w-full text-sm bg-background border-border focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200 [&::-webkit-search-cancel-button]:hidden"
                  value={searchTerm}
                  onChange={handleSearchChange}
                  type="search"
                />
              </div>
            </div>
          </div>
          
        
          {/* Table View */}
          <div className="border-t border-b border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr className="text-xs text-gray-500 h-8">
                  <th scope="col" className="px-3 py-2 text-left font-medium">Student</th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">ID</th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Program</th>
                  <th scope="col" className="px-3 py-2 text-left font-medium">Year & Section</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200 text-sm">
                {loading ? (
                <tr className="h-8">
                  <td colSpan={4} className="px-3 py-1 text-center">
                    <div className="flex justify-center">
                      <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                    </div>
                    <p className="mt-1 text-xs text-gray-500">Loading students...</p>
                  </td>
                </tr>
                ) : students.length === 0 ? (
                <tr className="h-8">
                  <td colSpan={4} className="px-3 py-1 text-center text-sm text-gray-500">
                    {pagination.totalCount === 0 
                      ? 'No students found. Add your first student!'
                      : 'No students match the current filters. Try adjusting your search or filters.'}
                    {pagination.totalCount > 0 && (
                      <Button 
                        variant="outline" 
                        size="sm"
                        className="mt-1 h-6 text-xs hover:bg-gray-100 hover:text-gray-900"
                        onClick={clearFilters}
                      >
                        Clear all filters
                      </Button>
                    )}
                  </td>
                </tr>
                ) : (
                  students.map((student) => (
                    <tr key={student.id} className="hover:bg-gray-50 h-8">
                      <td className="px-3 py-1 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {student.surname}, {student.firstname}{student.middlename ? ' ' + student.middlename.charAt(0) + '.' : ''}
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                        {student.student_id}
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                        <span className="truncate max-w-[120px] inline-block">{student.program}</span>
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
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
          </div>
        </div>
      </div>
      </PageWrapper>
    </Layout>
  );
}

export default Students;