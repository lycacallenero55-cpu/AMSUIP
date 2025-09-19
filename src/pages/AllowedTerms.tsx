import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar, Plus, Search, Trash2, Edit, Loader2, ChevronsUp, ChevronsDown } from "lucide-react";
import { format } from "date-fns";
import { useToast } from "@/components/ui/use-toast";
import { supabase } from "@/lib/supabase";
import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

 type AllowedTerm = {
  id: string;
  academic_year: string;
  semester: string;
  start_date: string; // ISO date
  end_date: string;   // ISO date
  created_at: string;
};

const AllowedTermsContent = () => {
  const { toast } = useToast();
  const [terms, setTerms] = useState<AllowedTerm[]>([]);
  // Sorting
  type TermsSortKey = 'year' | 'semester';
  const [sortKey, setSortKey] = useState<TermsSortKey>('year');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [loading, setLoading] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [pagination, setPagination] = useState({
    currentPage: 1,
    pageSize: 10,
    total: 0
  });
  const [displayPageSize, setDisplayPageSize] = useState(10);
  const [totalTermsCount, setTotalTermsCount] = useState(0);

  const [formData, setFormData] = useState<{
    academic_year: string;
    semester: string;
    start_date: string;
    end_date: string;
  }>({ academic_year: '', semester: '', start_date: '', end_date: '' });
  const [isFormOpen, setIsFormOpen] = useState<boolean>(false);
  const [editingTerm, setEditingTerm] = useState<AllowedTerm | null>(null);

  useEffect(() => {
    fetchAllowedTerms();
  }, []);

  const fetchAllowedTerms = async () => {
    try {
      setLoading(true);
      const { data, error, count } = await supabase
        .from('allowed_terms')
        .select('*', { count: 'exact' })
        .order('created_at', { ascending: false });
      if (error) throw error;
      setTerms(data || []);
      setTotalTermsCount(count || 0);
      setPagination(prev => ({ ...prev, total: count || 0 }));
    } catch (error) {
      console.error('Error fetching allowed terms:', error);
      toast({ title: 'Error', description: 'Failed to fetch allowed terms', variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  };

  const handlePageSizeChange = (newPageSize: number) => {
    const minPageSize = 10;
    const actualPageSize = Math.max(minPageSize, newPageSize);
    setPagination(prev => ({ ...prev, pageSize: actualPageSize, currentPage: 1 }));
    setDisplayPageSize(actualPageSize);
  };

  useEffect(() => {
    if (displayPageSize >= 999999) {
      setDisplayPageSize(totalTermsCount);
    }
  }, [totalTermsCount, displayPageSize]);

  const handleCreate = async () => {
    try {
      if (!formData.academic_year || !formData.semester || !formData.start_date || !formData.end_date) {
        toast({ title: 'Missing fields', description: 'Please fill in all fields.' });
        return;
      }

      if (editingTerm) {
        // Update existing term
        const { error } = await supabase
          .from('allowed_terms')
          .update({
            academic_year: formData.academic_year.trim(),
            semester: formData.semester.trim(),
            start_date: formData.start_date,
            end_date: formData.end_date,
          })
          .eq('id', editingTerm.id);
        if (error) throw error;
        toast({ title: 'Success', description: 'Allowed term updated successfully.' });
      } else {
        // Create new term
        const { error } = await supabase.from('allowed_terms').insert([
          {
            academic_year: formData.academic_year.trim(),
            semester: formData.semester.trim(),
            start_date: formData.start_date,
            end_date: formData.end_date,
          },
        ]);
        if (error) throw error;
        toast({ title: 'Success', description: 'Allowed term added successfully.' });
      }

      setFormData({ academic_year: '', semester: '', start_date: '', end_date: '' });
      setEditingTerm(null);
      setIsFormOpen(false);
      fetchAllowedTerms();
    } catch (error) {
      console.error('Error saving allowed term:', error);
      toast({ title: 'Error', description: 'Failed to save allowed term', variant: 'destructive' });
    }
  };

  const handleEdit = (term: AllowedTerm) => {
    setEditingTerm(term);
    setFormData({
      academic_year: term.academic_year,
      semester: term.semester,
      start_date: term.start_date,
      end_date: term.end_date,
    });
    setIsFormOpen(true);
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this allowed term?')) return;
    try {
      const { error } = await supabase.from('allowed_terms').delete().eq('id', id);
      if (error) throw error;
      toast({ title: 'Deleted', description: 'Allowed term removed.' });
      fetchAllowedTerms();
    } catch (error) {
      console.error('Delete error:', error);
      toast({ title: 'Error', description: 'Failed to delete allowed term', variant: 'destructive' });
    }
  };

  const formatDate = (iso: string) => format(new Date(iso), 'MMM d, yyyy');

  const filteredTerms = terms.filter((t) => {
    const hay = `${t.academic_year} ${t.semester}`.toLowerCase();
    return hay.includes(searchTerm.toLowerCase());
  });

  const startIndex = (pagination.currentPage - 1) * pagination.pageSize;
  const endIndex = startIndex + pagination.pageSize;
  const sortedTerms = [...filteredTerms].sort((a, b) => {
    const dir = sortDir === 'asc' ? 1 : -1;
    if (sortKey === 'year') return a.academic_year.localeCompare(b.academic_year) * dir;
    return a.semester.localeCompare(b.semester) * dir;
  });
  const paginatedTerms = sortedTerms.slice(startIndex, endIndex);
  const handleSort = (key: TermsSortKey) => {
    if (sortKey === key) setSortDir(prev => (prev === 'asc' ? 'desc' : 'asc'));
    else { setSortKey(key); setSortDir('asc'); }
  };

  return (
    <div className="px-6 py-4">
      <div className="mb-3">
        <div>
          <h1 className="text-2xl font-bold text-education-navy">ALLOWED TERMS</h1>
        </div>
      </div>
      
      {/* Big space between page title and card */}
      <div className="mb-16"></div>
      
      {/* Allowed Terms Section */}
      <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-base font-semibold text-education-navy">List of Allowed Terms</h3>
          <Button 
            variant="default"
            size="sm"
            className="h-8"
            onClick={() => {
              setEditingTerm(null);
              setFormData({ academic_year: '', semester: '', start_date: '', end_date: '' });
              setIsFormOpen(true);
            }}
          >
            <Plus className="w-4 h-4 mr-1" />
            Add Term
          </Button>
        </div>
        
        {/* Big space below List of Allowed Terms label */}
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
                placeholder="Search terms..."
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
              <tr className="text-xs text-black h-8">
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                  <div className="flex items-center gap-1">Academic Year
                    <button type="button" onClick={() => handleSort('year')} className="p-0.5 text-gray-500 hover:text-black">
                      {sortKey === 'year' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5 text-black"/> : <ChevronsDown className="w-3.5 h-3.5 text-black"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40 text-black"/>}
                    </button>
                  </div>
                </th>
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase">
                  <div className="flex items-center gap-1">Semester
                    <button type="button" onClick={() => handleSort('semester')} className="p-0.5 text-gray-500 hover:text-black">
                      {sortKey === 'semester' ? (sortDir === 'asc' ? <ChevronsUp className="w-3.5 h-3.5"/> : <ChevronsDown className="w-3.5 h-3.5"/>) : <ChevronsUp className="w-3.5 h-3.5 opacity-40"/>}
                    </button>
                  </div>
                </th>
                <th scope="col" className="px-3 py-2 text-left font-semibold uppercase"></th> {/* Empty for actions */}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200 text-xs text-gray-500">
              {loading ? null : paginatedTerms.length === 0 ? (
                <tr className="h-8">
                  <td colSpan={3} className="px-3 py-1 text-center text-sm text-gray-500">
                    {terms.length === 0 
                      ? 'No allowed terms found. Add your first term!'
                      : 'No terms match the current search. Try adjusting your search.'}
                  </td>
                </tr>
              ) : (
                <>
                  {paginatedTerms.map((term) => (
                    <tr key={term.id} className="hover:bg-gray-50 h-8">
                      <td className="px-3 py-1 whitespace-nowrap">
                        <div className="font-medium">{term.academic_year}</div>
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap">
                        {term.semester}
                      </td>
                      <td className="px-3 py-1 whitespace-nowrap text-right">
                        <div className="flex gap-1 justify-end">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => handleEdit(term)}
                          >
                            <Edit className="h-3 w-3 text-black" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => handleDelete(term.id)}
                          >
                            <Trash2 className="h-3 w-3 text-black" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {paginatedTerms.length < 10 && Array.from({ length: 10 - paginatedTerms.length }).map((_, idx) => (
                    <tr key={`filler-${idx}`} className="h-8">
                      <td colSpan={3} className="px-3 py-1">&nbsp;</td>
                    </tr>
                  ))}
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add/Edit Term Modal */}
      <Dialog open={isFormOpen} onOpenChange={(open) => {
        setIsFormOpen(open);
        if (!open) {
          setEditingTerm(null);
          setFormData({ academic_year: '', semester: '', start_date: '', end_date: '' });
        }
      }}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{editingTerm ? 'Edit Allowed Term' : 'New Allowed Term'}</DialogTitle>
            <DialogDescription>Provide academic year, semester, and date range.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="ay">Academic Year</Label>
              <Input
                id="ay"
                placeholder="e.g., 2024-2025"
                value={formData.academic_year}
                onChange={(e) => setFormData((p) => ({ ...p, academic_year: e.target.value }))}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label htmlFor="sem">Semester</Label>
              <Input
                id="sem"
                placeholder="e.g., 1st Semester"
                value={formData.semester}
                onChange={(e) => setFormData((p) => ({ ...p, semester: e.target.value }))}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-2">
                <Label htmlFor="start">Start Date</Label>
                <Input
                  id="start"
                  type="date"
                  value={formData.start_date}
                  onChange={(e) => setFormData((p) => ({ ...p, start_date: e.target.value }))}
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="end">End Date</Label>
                <Input
                  id="end"
                  type="date"
                  value={formData.end_date}
                  onChange={(e) => setFormData((p) => ({ ...p, end_date: e.target.value }))}
                />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsFormOpen(false)}>Cancel</Button>
            <Button 
              onClick={handleCreate}
              disabled={!formData.academic_year || !formData.semester || !formData.start_date || !formData.end_date}
            >
              {editingTerm ? 'Update' : 'Save'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

const AllowedTerms = () => {
  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <AllowedTermsContent />
      </PageWrapper>
    </Layout>
  );
};

export default AllowedTerms;