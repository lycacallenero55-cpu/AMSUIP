import React, { useState, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription } from "@/components/ui/dialog";
import { Download, FileSpreadsheet, FileText, Loader2, X, Check, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import * as XLSX from 'xlsx';
import Papa from 'papaparse';
import { supabase } from "@/lib/supabase";

interface StudentImportData {
  surname: string;
  middle_initial: string;
  firstname: string;
  student_id: string;
  program: string;
  year: string;
  section: string;
  sex: string;
  address: string;
  birthday: string;
  contact_no: string;
  email: string;
  [key: string]: string;
}

interface StudentImportProps {
  onImportComplete?: () => void;
  onImportSuccess?: () => void;
}

const StudentImport: React.FC<StudentImportProps> = ({ onImportComplete, onImportSuccess }) => {
  const [open, setOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [validation, setValidation] = useState<{
    total: number;
    valid: number;
    errors: string[];
  } | null>(null);
  const [previewData, setPreviewData] = useState<StudentImportData[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const requiredFields = [
    'surname',
    'firstname',
    'student_id',
    'program',
    'year',
    'section'
  ];

  type StudentRecord = Record<string, string | number | null | undefined>;

  const validateStudentData = (data: StudentRecord[]): { valid: boolean; errors: string[]; data: StudentImportData[] } => {
    const errors: string[] = [];
    const validData: StudentImportData[] = [];

    data.forEach((row, index) => {
      const rowNumber = index + 2; // +2 because of 0-based index and header row
      const rowErrors: string[] = [];
      const student: Partial<StudentImportData> = {};

      // Map and validate each field
      Object.entries(row).forEach(([key, value]) => {
        const normalizedKey = key.trim().toLowerCase().replace(/[^a-z0-9]/g, '_');
        student[normalizedKey] = String(value || '').trim();
      });

      // Check required fields
      requiredFields.forEach(field => {
        if (!student[field]) {
          rowErrors.push(`Row ${rowNumber}: Missing required field '${field}'`);
        }
      });

      // Validate year format if present
      if (student.year && !/^\d+(st|nd|rd|th)$/i.test(student.year)) {
        rowErrors.push(`Row ${rowNumber}: Invalid year format. Use format like '1st', '2nd', etc.`);
      }

      if (rowErrors.length === 0) {
        validData.push(student as StudentImportData);
      } else {
        errors.push(...rowErrors);
      }
    });

    return {
      valid: errors.length === 0,
      errors,
      data: validData
    };
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setValidation(null);
    setPreviewData([]);

    try {
      const fileType = file.name.split('.').pop()?.toLowerCase() || '';
      let data: StudentRecord[] = [];

      if (fileType === 'xlsx') {
        const arrayBuffer = await file.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer, { type: 'array' });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        data = XLSX.utils.sheet_to_json(worksheet);
      } else if (fileType === 'csv') {
        const text = await file.text();
        const result = Papa.parse(text, { header: true });
data = result.data as StudentRecord[];
      } else {
        throw new Error('Unsupported file type. Please upload a .xlsx or .csv file.');
      }

      // Filter out empty rows
      data = data.filter(row => 
        Object.values(row).some(value => 
          value !== null && value !== undefined && String(value).trim() !== ''
        )
      );

      const validationResult = validateStudentData(data);
      
      setPreviewData(validationResult.data);
      setValidation({
        total: data.length,
        valid: validationResult.data.length,
        errors: validationResult.errors
      });

    } catch (error) {
      console.error('Error processing file:', error);
      toast.error('Error processing file. Please check the file format and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleImport = async () => {
    if (!previewData || previewData.length === 0) {
      toast.error('No data to import');
      return;
    }
    
    setIsLoading(true);
    
    try {
      // Check if user is authenticated
      const { data: { user }, error: authError } = await supabase.auth.getUser();
      
      if (authError || !user) {
        throw new Error('You must be logged in to import students');
      }
      
      const { data: validData, errors } = validateStudentData(previewData);
      
      if (errors.length > 0) {
        setValidation({
          total: previewData.length,
          valid: validData.length,
          errors: errors.slice(0, 5) // Show first 5 errors
        });
        setIsLoading(false);
        return;
      }
      
      if (validData.length === 0) {
        toast.error('No valid data to import');
        setIsLoading(false);
        return;
      }

      // Prepare data for batch insert
      const studentsToImport = validData.map(student => ({
        ...student,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }));

      // Batch insert using Supabase with proper error handling
      const { data: insertedData, error: insertError } = await supabase
        .from('students')
        .insert(studentsToImport)
        .select()
        .select('*');

      if (insertError) {
        console.error('Supabase insert error:', insertError);
        throw new Error(insertError.message || 'Failed to import students');
      }

      if (insertedData && insertedData.length > 0) {
        const importedCount = insertedData.length;
        toast.success(`Successfully imported ${importedCount} students`);
        if (onImportSuccess) onImportSuccess();
        if (onImportComplete) onImportComplete();
        setOpen(false);
        setPreviewData([]);
        setFileName('');
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } else {
        throw new Error('No data was imported');
      }
    } catch (error) {
      console.error('Error importing students:', error);
      toast.error(`Error importing students: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setFileName('');
    setValidation(null);
    setPreviewData([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const downloadTemplate = () => {
    // Define the headers based on the StudentForm fields
    const headers = [
      'surname',
      'middle_initial',
      'firstname',
      'student_id',
      'program',
      'year',
      'section',
      'sex',
      'address',
      'birthday',
      'contact_no',
      'email'
    ];

    // Create a new workbook
    const wb = XLSX.utils.book_new();

    // Add a worksheet with the headers
    const ws = XLSX.utils.aoa_to_sheet([headers]);

    // Add some example data
    const exampleData = [
      {
        surname: 'Doe',
        middle_initial: 'M',
        firstname: 'John',
        student_id: '2023-001',
        program: 'Computer Science',
        year: '1st',
        section: 'BSCS 1A',
        sex: 'Male',
        address: '123 Main St',
        birthday: '2005-05-15',
        contact_no: '09123456789',
        email: 'john.doe@example.com'
      },
      {
        surname: 'Smith',
        middle_initial: 'A',
        firstname: 'Jane',
        student_id: '2023-002',
        program: 'Information Technology',
        year: '2nd',
        section: 'BSIT 2B',
        sex: 'Female',
        address: '456 Oak Ave',
        birthday: '2004-08-22',
        contact_no: '09123456780',
        email: 'jane.smith@example.com'
      }
    ];

    // Add the example data to the worksheet
    XLSX.utils.sheet_add_json(ws, exampleData, { header: headers, skipHeader: true, origin: 'A2' });

    // Add the worksheet to the workbook
    XLSX.utils.book_append_sheet(wb, ws, 'Students');

    // Generate the Excel file
    const excelBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });

    // Create a Blob and download link
    const blob = new Blob([excelBuffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'student_import_template.xlsx';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Dialog open={open} onOpenChange={(isOpen) => {
      if (!isOpen) resetForm();
      setOpen(isOpen);
    }}>
      <DialogTrigger asChild>
        <Button 
          variant="default"
          size="sm"
        >
          <Download className="w-4 h-4 mr-2" />
          Import
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold text-education-navy">
            Import Students
          </DialogTitle>
          <DialogDescription>
            Upload an Excel or CSV file to import multiple students at once.
            Download the template file to ensure proper formatting.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          <div className="border-2 border-dashed rounded-lg p-6 text-center">
            <div className="flex flex-col items-center justify-center space-y-4">
              <div className="p-3 rounded-full bg-accent/10">
                {fileName ? (
                  fileName.endsWith('.xlsx') ? (
                    <FileSpreadsheet className="w-8 h-8 text-accent" />
                  ) : (
                    <FileText className="w-8 h-8 text-accent" />
                  )
                ) : (
                  <Download className="w-8 h-8 text-muted-foreground" />
                )}
              </div>
              
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">
                  {fileName || 'Upload Excel (.xlsx) or CSV (.csv) file'}
                </p>
                <p className="text-xs text-muted-foreground">
                  The file should contain student data with appropriate headers. 
                  <button 
                    type="button" 
                    onClick={downloadTemplate}
                    className="text-primary hover:underline ml-1"
                  >
                    Download template
                  </button>
                </p>
              </div>

              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
              >
                {fileName ? 'Change File' : 'Select File'}
              </Button>
              
              <input
                ref={fileInputRef}
                type="file"
                accept=".xlsx,.csv"
                className="hidden"
                onChange={handleFileChange}
                disabled={isLoading}
              />
            </div>
          </div>

          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 mr-2 animate-spin" />
              <span>Processing file...</span>
            </div>
          )}

          {validation && (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                <div>
                  <h4 className="font-medium">File Analysis</h4>
                  <p className="text-sm text-muted-foreground">
                    {validation.valid} of {validation.total} students ready to import
                  </p>
                </div>
                {validation.errors.length > 0 ? (
                  <div className="flex items-center text-destructive">
                    <AlertCircle className="w-5 h-5 mr-2" />
                    <span>{validation.errors.length} issues found</span>
                  </div>
                ) : (
                  <div className="flex items-center text-green-600">
                    <Check className="w-5 h-5 mr-2" />
                    <span>All records are valid</span>
                  </div>
                )}
              </div>

              {validation.errors.length > 0 && (
                <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 max-h-48 overflow-y-auto">
                  <h4 className="font-medium text-destructive mb-2">Issues to fix</h4>
                  <ul className="text-sm space-y-1">
                    {validation.errors.slice(0, 10).map((error, index) => (
                      <li key={index} className="text-destructive">{error}</li>
                    ))}
                    {validation.errors.length > 10 && (
                      <li className="text-muted-foreground">
                        ...and {validation.errors.length - 10} more issues
                      </li>
                    )}
                  </ul>
                </div>
              )}

              {previewData.length > 0 && (
                <div className="border rounded-lg overflow-hidden">
                  <div className="bg-muted/50 p-2 border-b">
                    <h4 className="font-medium">Preview ({previewData.length} records)</h4>
                  </div>
                  <div className="max-h-60 overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-muted/25 border-b text-left">
                          <th className="p-2">ID</th>
                          <th className="p-2">Name</th>
                          <th className="p-2">Program</th>
                          <th className="p-2">Year</th>
                          <th className="p-2">Section</th>
                        </tr>
                      </thead>
                      <tbody>
                        {previewData.slice(0, 5).map((student, index) => (
                          <tr key={index} className="border-b hover:bg-muted/10">
                            <td className="p-2">{student.student_id}</td>
                            <td className="p-2">{`${student.surname}, ${student.firstname} ${student.middle_initial}`.trim()}</td>
                            <td className="p-2">{student.program}</td>
                            <td className="p-2">{student.year}</td>
                            <td className="p-2">{student.section}</td>
                          </tr>
                        ))}
                        {previewData.length > 5 && (
                          <tr>
                            <td colSpan={5} className="p-2 text-center text-muted-foreground text-sm">
                              ... and {previewData.length - 5} more records
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              <div className="flex justify-end space-x-2 pt-4">
                <Button
                  variant="outline"
                  onClick={() => resetForm()}
                  disabled={isLoading}
                >
                  Reset
                </Button>
                <Button
                  onClick={handleImport}
                  disabled={isLoading || previewData.length === 0}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Importing...
                    </>
                  ) : (
                    `Import ${previewData.length} Students`
                  )}
                </Button>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default StudentImport;
