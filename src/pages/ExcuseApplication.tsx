import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem } from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { useUnsavedChanges } from "@/hooks/useUnsavedChanges";
import { UnsavedChangesDialog } from "@/components/UnsavedChangesDialog";
import { Loader2 } from "lucide-react";
import PageWrapper from "@/components/PageWrapper";

import { FileText, Clock, AlertCircle, CheckCircle2, Plus, Search, Filter, Eye, Check, X, ChevronsUpDown, CalendarIcon, Edit, ZoomIn, ZoomOut, Trash2 } from "lucide-react";
import { format } from "date-fns";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/lib/supabase";
import Layout from "@/components/Layout";
import { cn } from "@/lib/utils";

type ExcuseStatus = 'pending' | 'approved' | 'rejected';

type ExcuseApplication = {
  id: number;
  student_id: number;
  session_id?: number;
  absence_date: string;
  
  documentation_url?: string;
  status: ExcuseStatus;
  reviewed_by?: string;
  reviewed_at?: string;
  review_notes?: string;
  created_at: string;
  updated_at: string;
  // Related data
  students?: {
    id: number;
    firstname: string;
    surname: string;
    student_id: string;
    program: string;
    year: string;
    section: string;
  };
  sessions?: {
    id: number;
    title: string;
    date: string;
  };
};

type ExcuseFormData = {
  student_id: string;
  session_id?: string;
  absence_date: string;
  excuse_image?: File;
  documentation_url?: string;
};

const ExcuseApplicationContent = () => {
  const { toast } = useToast();
  const [excuses, setExcuses] = useState<ExcuseApplication[]>([]);
  const [loading, setLoading] = useState(true);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [isViewOpen, setIsViewOpen] = useState(false);
  const [selectedExcuse, setSelectedExcuse] = useState<ExcuseApplication | null>(null);
  const [formData, setFormData] = useState<ExcuseFormData>({
    student_id: '',
    absence_date: '', // Keep for type compatibility but won't be used
  });
  const [students, setStudents] = useState<any[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [openStudentSelect, setOpenStudentSelect] = useState(false);
  const [openSessionSelect, setOpenSessionSelect] = useState(false);
  const [imageZoom, setImageZoom] = useState(1);
  const [imagePan, setImagePan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isEditMode, setIsEditMode] = useState(false);
  const [viewMode, setViewMode] = useState<'view' | 'edit'>('view');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<number | null>(null);
  const [displayPageSize, setDisplayPageSize] = useState(10);
  const [totalExcusesCount, setTotalExcusesCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');

  const {
    showConfirmDialog,
    markAsChanged,
    markAsSaved,
    handleClose,
    confirmClose,
    cancelClose,
    handleOpenChange,
  } = useUnsavedChanges({
    onClose: () => {
      setIsFormOpen(false);
      setIsEditMode(false);
      setSelectedExcuse(null);
      setFormData({ 
        student_id: '', 
        session_id: '',
        absence_date: '',
        documentation_url: ''
      });
    },
    enabled: isFormOpen,
  });

  useEffect(() => {
    fetchExcuses();
    fetchStudents();
    fetchSessions();
  }, []);

  const fetchExcuses = async () => {
    try {
      setLoading(true);
      const { data, error } = await supabase
        .from('excuse_applications')
        .select(`
          *,
          students!student_id (
            id,
            firstname,
            surname,
            student_id,
            program,
            year,
            section
          ),
          sessions!session_id (
            id,
            title,
            date
          )
        `)
        .order('created_at', { ascending: false });

      if (error) throw error;
      console.log('Fetched excuses data:', data);
      const excusesData = (data || []).map(excuse => ({
        ...excuse,
        status: excuse.status as ExcuseStatus
      }));
      setExcuses(excusesData);
      setTotalExcusesCount(excusesData.length);
    } catch (error) {
      console.error('Error fetching excuses:', error);
      toast({
        title: "Error",
        description: "Failed to fetch excuse applications",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle page size change
  const handlePageSizeChange = (newPageSize: number) => {
    // Allow very large numbers for "ALL" case, otherwise ensure minimum value of 10
    const validPageSize = newPageSize >= 999999 ? newPageSize : Math.max(10, newPageSize);
    
    // Update display page size (what user sees in the control)
    if (newPageSize >= 999999) {
      setDisplayPageSize(totalExcusesCount);
    } else {
      setDisplayPageSize(validPageSize);
    }
  };

  // Update display page size when total excuses count changes (for "ALL" case)
  useEffect(() => {
    if (displayPageSize >= 999999) {
      setDisplayPageSize(totalExcusesCount);
    }
  }, [totalExcusesCount, displayPageSize]);

  const fetchStudents = async () => {
    try {
      const { data, error } = await supabase
        .from('students')
        .select('id, firstname, surname, student_id, program, year, section')
        .order('firstname');

      if (error) throw error;
      setStudents(data || []);
    } catch (error) {
      console.error('Error fetching students:', error);
    }
  };

  const fetchSessions = async () => {
    try {
      const { data, error } = await supabase
        .from('sessions')
        .select('id, title, date')
        .order('date', { ascending: false });

      if (error) throw error;
      setSessions(data || []);
    } catch (error) {
      console.error('Error fetching sessions:', error);
    }
  };

  const handleSubmitExcuse = async () => {
    try {
      let excuse_image_url = null;
      
      // Upload image if provided
      if (formData.excuse_image) {
        const fileExt = formData.excuse_image.name.split('.').pop();
        const fileName = `${Date.now()}-${Math.random().toString(36).substring(2, 15)}.${fileExt}`;
        const filePath = `excuse-letters/${fileName}`;
        
        const { error: uploadError } = await supabase.storage
          .from('excuse-letters')
          .upload(filePath, formData.excuse_image);
          
        if (uploadError) throw uploadError;
        
        const { data: { publicUrl } } = supabase.storage
          .from('excuse-letters')
          .getPublicUrl(filePath);
          
        excuse_image_url = publicUrl;
      }

      if (isEditMode && selectedExcuse) {
        // Update existing excuse
        const { error } = await supabase
          .from('excuse_applications')
          .update({
            student_id: parseInt(formData.student_id),
            session_id: formData.session_id ? parseInt(formData.session_id) : null,
            absence_date: formData.absence_date,
            documentation_url: excuse_image_url || formData.documentation_url,
            updated_at: new Date().toISOString()
          })
        .eq('id', selectedExcuse.id);

        if (error) throw error;

        toast({
          title: "Success",
          description: "Excuse application updated successfully",
        });
      } else {
        // Create new excuse
        const { error } = await supabase
          .from('excuse_applications')
          .insert([{
            student_id: parseInt(formData.student_id),
            session_id: formData.session_id ? parseInt(formData.session_id) : null,
            absence_date: formData.absence_date || new Date().toISOString().split('T')[0],
            documentation_url: excuse_image_url || formData.documentation_url,
            status: 'pending'
          }]);

        if (error) throw error;

        toast({
          title: "Success",
          description: "Excuse application submitted successfully",
        });
      }

      markAsSaved();
      setIsFormOpen(false);
      setIsEditMode(false);
      setSelectedExcuse(null);
      setFormData({
        student_id: '',
        absence_date: '',
      });
      fetchExcuses();
    } catch (error) {
      console.error('Error submitting excuse:', error);
      toast({
        title: "Error",
        description: isEditMode ? "Failed to update excuse application" : "Failed to submit excuse application",
        variant: "destructive",
      });
    }
  };

  const handleUpdateStatus = async (id: number, status: ExcuseStatus, notes?: string) => {
    try {
      const { error } = await supabase
        .from('excuse_applications')
        .update({
          status,
          review_notes: notes,
          reviewed_at: new Date().toISOString(),
        })
        .eq('id', id);

      if (error) throw error;

      toast({
        title: "Success",
        description: `Excuse application ${status}`,
      });

      fetchExcuses();
      setIsViewOpen(false);
    } catch (error) {
      console.error('Error updating status:', error);
      toast({
        title: "Error",
        description: "Failed to update status",
        variant: "destructive",
      });
    }
  };

  const handleDeleteExcuse = async (id: number) => {
    try {
      console.log('Deleting excuse with ID:', id, 'Type:', typeof id);
      
      // First, let's check what the actual ID looks like in the database
      const { data: checkData, error: checkError } = await supabase
        .from('excuse_applications')
        .select('id')
        .limit(1);
      
      if (checkError) {
        console.error('Error checking ID format:', checkError);
      } else {
        console.log('Sample ID from database:', checkData?.[0]?.id, 'Type:', typeof checkData?.[0]?.id);
      }
      
      // Try the delete operation
      const { error, count } = await supabase
        .from('excuse_applications')
        .delete()
        .eq('id', id);

      console.log('Delete result - Error:', error, 'Count:', count);
      
      // Also try to fetch the record to see if it exists
      const { data: checkRecord, error: checkRecordError } = await supabase
        .from('excuse_applications')
        .select('*')
        .eq('id', id)
        .single();
      
      console.log('Record check - Data:', checkRecord, 'Error:', checkRecordError);

      if (error) {
        console.error('Supabase delete error:', error);
        throw error;
      }

      if (count === 0) {
        console.log('No records were deleted, trying alternative approach...');
        
        // Try alternative delete approach
        const { error: altError, count: altCount } = await supabase
          .from('excuse_applications')
          .delete()
          .eq('id', id);
        
        console.log('Alternative delete result - Error:', altError, 'Count:', altCount);
        
        if (altCount === 0) {
          toast({
            title: "Warning",
            description: "No record found to delete",
            variant: "destructive",
          });
          return;
        }
      }

      console.log('Delete successful, refreshing list...');
      
      toast({
        title: "Success",
        description: "Excuse application deleted successfully",
      });

      // Refresh the list immediately
      await fetchExcuses();
      setShowDeleteConfirm(false);
      setDeleteTarget(null);
    } catch (error) {
      console.error('Error deleting excuse:', error);
      toast({
        title: "Error",
        description: "Failed to delete excuse application",
        variant: "destructive",
      });
    }
  };

  const handleImageMouseDown = (e: React.MouseEvent) => {
    if (imageZoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - imagePan.x, y: e.clientY - imagePan.y });
    }
  };

  const handleImageMouseMove = (e: React.MouseEvent) => {
    if (isDragging && imageZoom > 1) {
      // Add drag sensitivity control - reduce movement by dividing by zoom level
      const sensitivity = 1 / imageZoom;
      setImagePan({
        x: (e.clientX - dragStart.x) * sensitivity,
        y: (e.clientY - dragStart.y) * sensitivity
      });
    }
  };

  const handleImageMouseUp = () => {
    setIsDragging(false);
  };

  // Reset pan position when zoom changes
  const handleZoomChange = (newZoom: number) => {
    setImageZoom(newZoom);
    // Reset pan position when zoom returns to 100% or below
    if (newZoom <= 1) {
      setImagePan({ x: 0, y: 0 });
    }
  };

  const getStatusDisplay = (status: ExcuseStatus) => {
    return (
      <span className="text-sm text-gray-500 capitalize">
        {status}
      </span>
    );
  };

  // Filter excuses based on search term
  const filteredExcuses = excuses.filter(excuse => {
    const matchesSearch = 
      `${excuse.students?.firstname} ${excuse.students?.surname}`.toLowerCase().includes(searchTerm.toLowerCase()) ||
      excuse.students?.student_id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      excuse.students?.program?.toLowerCase().includes(searchTerm.toLowerCase());
    
    return matchesSearch;
  });

  return (
    <div className="px-6 py-4">
      <div className="mb-3">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-1">
          <div>
            <h1 className="text-2xl font-bold text-education-navy">EXCUSE APPLICATIONS</h1>
            <p className="text-sm text-muted-foreground">
              Review and manage student excuse applications for absences
            </p>
          </div>
          <div className="mt-2 md:mt-0">
            <Button 
              className="bg-gradient-primary shadow-glow h-9"
              onClick={() => setIsFormOpen(true)}
            >
              <Plus className="w-4 h-4 mr-2" />
              New Application
            </Button>
          </div>
        </div>
      </div>
      
      {/* Excuse Applications Section */}
      <div className="bg-white rounded-lg shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-base font-semibold text-education-navy">List of Excuse Applications</h3>
        </div>
        
        {/* Big space below List of Excuse Applications label */}
        <div className="mb-8"></div>
        
        {/* Top controls row */}
        <div className="flex items-center justify-between gap-4 p-0 mb-3">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600">Showed:</span>
            <Select
              value={displayPageSize >= 999999 ? "all" : displayPageSize.toString()}
              onValueChange={(value) => {
                if (value === "all") {
                  handlePageSizeChange(999999);
                } else {
                  handlePageSizeChange(parseInt(value));
                }
              }}
            >
              <SelectTrigger className="h-8 w-24">
                <SelectValue>
                  {displayPageSize.toString()}
                </SelectValue>
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
            <span className="text-sm text-gray-600">Search:</span>
            <div className="relative min-w-[240px] max-w-[340px]">
              <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <Input
                placeholder="Search applications..."
                className="pl-7 pr-7 h-8 w-full text-sm bg-background border-border focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
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
                <th scope="col" className="px-3 py-2 text-left font-medium">Name</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">ID</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">Date of Session</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">Documentation</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">Status</th>
                <th scope="col" className="px-3 py-2 text-left font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200 text-sm">
              {loading ? (
                <tr className="h-8">
                  <td colSpan={6} className="px-3 py-1 text-center">
                    <div className="flex justify-center">
                      <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                    </div>
                    <p className="mt-1 text-xs text-gray-500">Loading applications...</p>
                  </td>
                </tr>
              ) : filteredExcuses.length === 0 ? (
                <tr className="h-8">
                  <td colSpan={6} className="px-3 py-1 text-center text-sm text-gray-500">
                    {excuses.length === 0 
                      ? 'No excuse applications found. Add your first application!'
                      : 'No applications match the current search. Try adjusting your search.'}
                  </td>
                </tr>
              ) : (
                filteredExcuses.map((excuse) => (
                  <tr key={excuse.id} className="hover:bg-gray-50 h-8">
                    <td className="px-3 py-1 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">
                        {excuse.students?.firstname} {excuse.students?.surname}
                      </div>
                    </td>
                    <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                      {excuse.students?.student_id}
                    </td>
                    <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                      {format(new Date(excuse.absence_date), 'MMM d, yyyy')}
                    </td>
                    <td className="px-3 py-1 whitespace-nowrap">
                      {excuse.documentation_url ? (
                        <div 
                          className="w-12 h-8 bg-gray-100 rounded border cursor-pointer overflow-hidden"
                          onClick={() => {
                            setSelectedExcuse(excuse);
                            setViewMode('view');
                            setIsViewOpen(true);
                          }}
                        >
                          <img 
                            src={excuse.documentation_url} 
                            alt="Excuse letter preview" 
                            className="w-full h-full object-cover hover:opacity-80 transition-opacity"
                          />
                        </div>
                      ) : (
                        <span className="text-sm text-gray-400">No attachment</span>
                      )}
                    </td>
                    <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                      {getStatusDisplay(excuse.status)}
                    </td>
                    <td className="px-3 py-1 whitespace-nowrap text-right">
                      <div className="flex gap-1 justify-end">
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-6 w-6 p-0"
                          onClick={() => {
                            setSelectedExcuse(excuse);
                            setViewMode('view');
                            setIsViewOpen(true);
                          }}
                        >
                          <Eye className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-6 w-6 p-0"
                          onClick={() => {
                            setSelectedExcuse(excuse);
                            setViewMode('edit');
                            setIsEditMode(true);
                            setFormData({
                              student_id: excuse.student_id?.toString() || '',
                              session_id: excuse.session_id?.toString() || '',
                              absence_date: excuse.absence_date || '',
                              documentation_url: excuse.documentation_url || ''
                            });
                            setIsFormOpen(true);
                          }}
                        >
                          <Edit className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-6 w-6 p-0"
                          onClick={() => {
                            setDeleteTarget(excuse.id);
                            setShowDeleteConfirm(true);
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Form Dialog */}
      <Dialog open={isFormOpen} onOpenChange={handleOpenChange}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{isEditMode ? 'Edit Excuse Application' : 'New Excuse Application'}</DialogTitle>
            <DialogDescription>
              {isEditMode ? 'Update the excuse application details.' : 'Submit a new excuse application for a student absence.'}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="student">Student</Label>
              <Popover open={openStudentSelect} onOpenChange={setOpenStudentSelect}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={openStudentSelect}
                    className="w-full justify-between"
                  >
                    {formData.student_id
                      ? students.find((student) => student.id.toString() === formData.student_id)?.firstname + ' ' + students.find((student) => student.id.toString() === formData.student_id)?.surname + ' (' + students.find((student) => student.id.toString() === formData.student_id)?.student_id + ')'
                      : "Select student..."}
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-full p-0 z-[100]">
                  <Command>
                    <CommandInput placeholder="Search students..." />
                    <CommandEmpty>No student found.</CommandEmpty>
                    <CommandGroup>
                      {students.map((student) => (
                        <CommandItem
                          key={student.id}
                           onSelect={() => {
                             setFormData(prev => ({ ...prev, student_id: student.id.toString() }));
                             setOpenStudentSelect(false);
                             markAsChanged();
                           }}
                        >
                          <Check
                            className={cn(
                              "mr-2 h-4 w-4",
                              formData.student_id === student.id.toString() ? "opacity-100" : "opacity-0"
                            )}
                          />
                          {student.firstname} {student.surname} ({student.student_id})
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>

            <div>
              <Label htmlFor="session">Session *</Label>
              <Popover open={openSessionSelect} onOpenChange={setOpenSessionSelect}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={openSessionSelect}
                    className="w-full justify-between"
                  >
                    {formData.session_id
                      ? sessions.find((session) => session.id.toString() === formData.session_id)?.title + ' - ' + format(new Date(sessions.find((session) => session.id.toString() === formData.session_id)?.date), 'MMM d, yyyy')
                      : "Select session..."}
                    <div className="flex items-center gap-1">
                      <CalendarIcon className="h-4 w-4 shrink-0 opacity-50" />
                      <ChevronsUpDown className="h-4 w-4 shrink-0 opacity-50" />
                    </div>
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-full p-0 z-[100]">
                  <Command>
                    <CommandInput placeholder="Search sessions or dates..." />
                    <CommandEmpty>No session found.</CommandEmpty>
                    <CommandGroup>
                      {sessions.map((session) => (
                        <CommandItem
                          key={session.id}
                           onSelect={() => {
                             setFormData(prev => ({ ...prev, session_id: session.id.toString() }));
                             setOpenSessionSelect(false);
                             markAsChanged();
                           }}
                        >
                          <Check
                            className={cn(
                              "mr-2 h-4 w-4",
                              formData.session_id === session.id.toString() ? "opacity-100" : "opacity-0"
                            )}
                          />
                          <div className="flex flex-col">
                            <span>{session.title}</span>
                            <span className="text-sm text-muted-foreground">
                              {format(new Date(session.date), 'EEEE, MMM d, yyyy')}
                            </span>
                          </div>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>


            <div>
              <Label htmlFor="excuse-image">Handwritten Excuse Letter</Label>
              <Input
                id="excuse-image"
                type="file"
                accept="image/*"
                 onChange={(e) => {
                   const file = e.target.files?.[0];
                   if (file) {
                     setFormData(prev => ({ ...prev, excuse_image: file }));
                     markAsChanged();
                   }
                 }}
                className="cursor-pointer"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Please attach a clear photo of your handwritten excuse letter
              </p>
            </div>


          </div>
          <DialogFooter>
            <Button variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button 
              onClick={handleSubmitExcuse}
              disabled={!formData.student_id || !formData.session_id || !formData.excuse_image}
            >
              Submit Application
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <UnsavedChangesDialog
        open={showConfirmDialog}
        onConfirm={confirmClose}
        onCancel={cancelClose}
      />

      {/* View Dialog */}
      <Dialog open={isViewOpen} onOpenChange={(open) => {
        setIsViewOpen(open);
        if (!open) {
          setImageZoom(1);
          setImagePan({ x: 0, y: 0 });
          setIsDragging(false);
        }
      }}>
        <DialogContent className="max-w-7xl w-full h-[85vh] flex flex-col">
          <DialogHeader className="flex-shrink-0">
            <DialogTitle>Excuse Application Details</DialogTitle>
          </DialogHeader>
          {selectedExcuse && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 flex-1 overflow-hidden">
              {/* Left side - Application details */}
              <div className="lg:col-span-1 bg-muted/30 p-6 rounded-lg border space-y-6">
                {/* Header Section */}
                <div className="border-b border-border pb-4">
                  <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    Application Details
                  </h3>
                </div>
                
                {/* Student Information */}
                <div className="space-y-3">
                  <div className="flex items-start gap-3 p-3 bg-background rounded-md border border-border/50">
                    <div className="p-2 bg-primary/10 rounded-full">
                      <svg className="h-4 w-4 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Student</Label>
                      <p className="text-base font-semibold text-foreground mt-1">
                        {selectedExcuse.students?.firstname || 'Unknown'} {selectedExcuse.students?.surname || 'Student'}
                      </p>
                      <p className="text-sm text-muted-foreground flex items-center gap-2 mt-1">
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-secondary text-secondary-foreground">
                          {selectedExcuse.students?.student_id || 'N/A'}
                        </span>
                        <span className="text-muted-foreground">•</span>
                        <span>{selectedExcuse.students?.program || 'N/A'}</span>
                      </p>
                    </div>
                  </div>
                </div>
                
                {/* Status Section */}
                <div className="space-y-3">
                  <div className="flex items-start gap-3 p-3 bg-background rounded-md border border-border/50">
                    <div className="p-2 bg-primary/10 rounded-full">
                      {selectedExcuse.status === 'approved' ? (
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                      ) : selectedExcuse.status === 'rejected' ? (
                        <AlertCircle className="h-4 w-4 text-red-600" />
                      ) : (
                        <Clock className="h-4 w-4 text-yellow-600" />
                      )}
                    </div>
                    <div className="flex-1">
                      <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Status</Label>
                      <div className="mt-2">
                        {getStatusBadge(selectedExcuse.status)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Absence Information */}
                <div className="space-y-3">
                  <div className="flex items-start gap-3 p-3 bg-background rounded-md border border-border/50">
                    <div className="p-2 bg-primary/10 rounded-full">
                      <CalendarIcon className="h-4 w-4 text-primary" />
                    </div>
                    <div className="flex-1">
                      <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Absence Date</Label>
                      <p className="text-base font-medium text-foreground mt-1">
                        {format(new Date(selectedExcuse.absence_date), 'EEEE, MMMM d, yyyy')}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Review Notes Section */}
                {selectedExcuse.review_notes && (
                  <div className="space-y-3">
                    <div className="p-3 bg-background rounded-md border border-border/50">
                      <div className="flex items-start gap-3 mb-3">
                        <div className="p-2 bg-primary/10 rounded-full">
                          <svg className="h-4 w-4 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Review Notes</Label>
                        </div>
                      </div>
                      <div className="pl-11">
                        <p className="text-sm text-foreground whitespace-pre-wrap leading-relaxed bg-muted/50 p-3 rounded border-l-2 border-primary/20">
                          {selectedExcuse.review_notes}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Timestamps Section */}
                <div className="space-y-3 pt-4 border-t border-border/50">
                  <div className="flex items-start gap-3 p-3 bg-background rounded-md border border-border/50">
                    <div className="p-2 bg-primary/10 rounded-full">
                      <Clock className="h-4 w-4 text-primary" />
                    </div>
                    <div className="flex-1">
                      <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Timeline</Label>
                      <div className="mt-2 space-y-2">
                        <div className="flex items-center gap-2 text-sm">
                          <div className="w-2 h-2 bg-primary rounded-full"></div>
                          <span className="font-medium text-foreground">Submitted:</span>
                          <span className="text-muted-foreground">{format(new Date(selectedExcuse.created_at), 'MMM d, yyyy h:mm a')}</span>
                        </div>
                        {selectedExcuse.reviewed_at && (
                          <div className="flex items-center gap-2 text-sm">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="font-medium text-foreground">Reviewed:</span>
                            <span className="text-muted-foreground">{format(new Date(selectedExcuse.reviewed_at), 'MMM d, yyyy h:mm a')}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Actions Section */}
                {selectedExcuse.status === 'pending' && (
                  <div className="pt-4 border-t border-border/50">
                    <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3 block">Actions</Label>
                    <div className="flex gap-3">
                      <Button
                        variant="outline"
                        className="flex-1 text-green-600 border-green-200 hover:bg-green-50 transition-colors"
                        onClick={() => handleUpdateStatus(selectedExcuse.id, 'approved')}
                      >
                        <Check className="h-4 w-4 mr-2" />
                        Approve
                      </Button>
                      <Button
                        variant="outline"
                        className="flex-1 text-red-600 border-red-200 hover:bg-red-50 transition-colors"
                        onClick={() => handleUpdateStatus(selectedExcuse.id, 'rejected')}
                      >
                        <X className="h-4 w-4 mr-2" />
                        Reject
                      </Button>
                    </div>
                  </div>
                )}
              </div>

              {/* Right side - Image with zoom controls */}
              <div className="lg:col-span-2 flex flex-col space-y-4">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium">Excuse Letter</Label>
                  {selectedExcuse.documentation_url && (
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleZoomChange(Math.max(0.5, imageZoom - 0.25))}
                      >
                        <ZoomOut className="h-4 w-4" />
                      </Button>
                      <span className="text-xs text-muted-foreground">{Math.round(imageZoom * 100)}%</span>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleZoomChange(Math.min(3, imageZoom + 0.25))}
                      >
                        <ZoomIn className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleZoomChange(1)}
                        className="ml-2"
                      >
                        Reset
                      </Button>
                    </div>
                  )}
                </div>
                
                {selectedExcuse.documentation_url ? (
                  <div 
                    className="border rounded-lg overflow-hidden flex-1 bg-gray-50 relative cursor-grab active:cursor-grabbing"
                    onMouseDown={handleImageMouseDown}
                    onMouseMove={handleImageMouseMove}
                    onMouseUp={handleImageMouseUp}
                    onMouseLeave={handleImageMouseUp}
                  >
                    <img 
                      src={selectedExcuse.documentation_url} 
                      alt="Excuse letter" 
                      className="transition-transform duration-200 max-w-none object-contain w-full h-full"
                      style={{ 
                        transform: `scale(${imageZoom}) translate(${imagePan.x}px, ${imagePan.y}px)`,
                        transformOrigin: 'center center'
                      }}
                      draggable={false}
                    />
                  </div>
                ) : (
                  <p className="text-sm text-gray-400">No excuse letter attached</p>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Excuse Application</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this excuse application? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteConfirm(false)}>
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={() => deleteTarget && handleDeleteExcuse(deleteTarget)}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

    </div>
  );
};

const ExcuseApplication = () => {
  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <ExcuseApplicationContent />
      </PageWrapper>
    </Layout>
  );
};

export default ExcuseApplication;