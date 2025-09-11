import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2 } from "lucide-react";

import { 
  Search, 
  CheckCircle2,
  XCircle,
  UserCheck,
  UserX,
  Clock
} from "lucide-react";
import Layout from "@/components/Layout";
import PageWrapper from "@/components/PageWrapper";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";
import { format } from "date-fns";

type AccountStatus = 'active' | 'inactive' | 'pending' | 'suspended';
type UserRole = 'admin' | 'Instructor' | 'SSG officer' | 'ROTC admin' | 'ROTC officer';

interface Profile {
  id: string;
  email: string;
  first_name: string | null;
  last_name: string | null;
  role: UserRole;
  status: AccountStatus;
  created_at: string;
  approved_at: string | null;
  rejected_at: string | null;
}

// Role Component
const RoleDisplay = ({ role }: { role: UserRole }) => {
  return (
    <span className="text-sm text-gray-500">
      {role.charAt(0).toUpperCase() + role.slice(1)}
    </span>
  );
};

// Status Component
const StatusDisplay = ({ status }: { status: AccountStatus }) => {
  return (
    <span className="text-sm text-gray-500 capitalize">
      {status}
    </span>
  );
};

const Accounts = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRole, setSelectedRole] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [currentUserProfile, setCurrentUserProfile] = useState<Profile | null>(null);
  const [displayPageSize, setDisplayPageSize] = useState(10);
  const [totalAccountsCount, setTotalAccountsCount] = useState(0);
  const { user } = useAuth();
  


  // Load current user profile and determine if admin
  const loadCurrentUserProfile = async () => {
    if (!user) return;
    
    try {
      // Try admin first
      const { data: adminData } = await supabase
        .from('admin')
        .select('*')
        .eq('id', user.id)
        .maybeSingle();
      
      let data: any = adminData;
      if (adminData) {
        // Explicitly set role to 'admin' for admin users
        data = { ...adminData, role: 'admin' };
      } else {
        const { data: userData } = await supabase
          .from('users')
          .select('*')
          .eq('id', user.id)
          .maybeSingle();
        data = userData;
      }

      setCurrentUserProfile(data);
      
      // If user is admin, load all profiles, otherwise just set their own
      if (adminData) {
        loadAllAccounts();
      } else {
        setProfiles([data]);
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Error loading user profile:', error);
      toast.error('Failed to load profile');
      setIsLoading(false);
    }
  };

  // Load all accounts (only for admins)
  const loadAllAccounts = async () => {
    try {
      const { data: admins } = await supabase
        .from('admin')
        .select('*')
        .order('created_at', { ascending: false });
      const { data: usersRows } = await supabase
        .from('users')
        .select('*')
        .order('created_at', { ascending: false });

      // Explicitly set role to 'admin' for admin users and ensure proper typing
      const adminProfiles = (admins || []).map(admin => ({ 
        ...admin, 
        role: 'admin' as UserRole,
        status: (admin.status || 'active') as AccountStatus
      }));
      const userProfiles = (usersRows || []).map(user => ({ 
        ...user, 
        role: user.role as UserRole,
        status: (user.status || 'active') as AccountStatus
      }));
      
      const allProfiles = [...adminProfiles, ...userProfiles];
      setProfiles(allProfiles);
      setTotalAccountsCount(allProfiles.length);
    } catch (error) {
      console.error('Error loading accounts:', error);
      toast.error('Failed to load accounts');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle page size change
  const handlePageSizeChange = (newPageSize: number) => {
    // Allow very large numbers for "ALL" case, otherwise ensure minimum value of 10
    const validPageSize = newPageSize >= 999999 ? newPageSize : Math.max(10, newPageSize);
    
    // Update display page size (what user sees in the control)
    if (newPageSize >= 999999) {
      setDisplayPageSize(totalAccountsCount);
    } else {
      setDisplayPageSize(validPageSize);
    }
  };

  // Update display page size when total accounts count changes (for "ALL" case)
  useEffect(() => {
    if (displayPageSize >= 999999) {
      setDisplayPageSize(totalAccountsCount);
    }
  }, [totalAccountsCount, displayPageSize]);

  useEffect(() => {
    if (user) {
      loadCurrentUserProfile();
    }
  }, [user]);

  // Approve user
  const handleApprove = async (userId: string) => {
    try {
      const { error } = await supabase.rpc('approve_user', {
        user_id: userId,
        approver_id: user?.id
      });

      if (error) throw error;
      
      toast.success('User approved successfully');
      if (currentUserProfile?.role === 'admin') {
        loadAllAccounts();
      }
    } catch (error) {
      console.error('Error approving user:', error);
      toast.error('Failed to approve user');
    }
  };

  // Reject user
  const handleReject = async (userId: string) => {
    try {
      const { error } = await supabase.rpc('reject_user', {
        user_id: userId,
        rejector_id: user?.id
      });

      if (error) throw error;
      
      toast.success('User rejected successfully');
      if (currentUserProfile?.role === 'admin') {
        loadAllAccounts();
      }
    } catch (error) {
      console.error('Error rejecting user:', error);
      toast.error('Failed to reject user');
    }
  };



  // Filter profiles
  const filteredProfiles = profiles.filter(profile => {
    const matchesSearch = 
      `${profile.first_name} ${profile.last_name}`.toLowerCase().includes(searchTerm.toLowerCase()) ||
      profile.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRole = selectedRole === 'all' || profile.role === selectedRole;
    const matchesStatus = selectedStatus === 'all' || profile.status === selectedStatus;
    
    return matchesSearch && matchesRole && matchesStatus;
  });

  if (isLoading) {
    return (
      <Layout>
        <div className="container mx-auto p-4">
          <div className="text-center">Loading accounts...</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <PageWrapper skeletonType="table">
        <div className="px-6 py-4 space-y-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-lg font-bold tracking-tight">ACCOUNT MANAGEMENT</h1>
            <p className="text-sm text-muted-foreground">
              Manage user accounts and access permissions
            </p>
          </div>

        </div>

        {/* Show accounts table only for admins */}
        {currentUserProfile?.role === 'admin' ? (
          <div className="bg-white rounded-lg shadow-sm p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-base font-semibold text-education-navy">List of Accounts</h3>
            </div>
            
            {/* Big space below List of Accounts label */}
            <div className="mb-8"></div>
            
            {/* Show search and filters inside the card */}
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
                    type="search"
                    placeholder="Search accounts..."
                    className="pl-7 pr-7 h-8 w-full text-sm bg-background border-border focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all duration-200"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </div>
              </div>
            </div>
            
            <div className="border-t border-b border-gray-200 overflow-hidden">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr className="text-xs text-gray-500 h-8">
                      <th scope="col" className="px-3 py-2 text-left font-medium">User</th>
                      <th scope="col" className="px-3 py-2 text-left font-medium">Email</th>
                      <th scope="col" className="px-3 py-2 text-left font-medium">Role</th>
                      <th scope="col" className="px-3 py-2 text-left font-medium">Status</th>
                      <th scope="col" className="px-3 py-2 text-left font-medium">Created</th>
                      <th scope="col" className="px-3 py-2 text-left font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200 text-sm">
                    {isLoading ? (
                      <tr className="h-8">
                        <td colSpan={6} className="px-3 py-1 text-center">
                          <div className="flex justify-center">
                            <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                          </div>
                          <p className="mt-1 text-xs text-gray-500">Loading accounts...</p>
                        </td>
                      </tr>
                    ) : filteredProfiles.length === 0 ? (
                      <tr className="h-8">
                        <td colSpan={6} className="px-3 py-1 text-center text-sm text-gray-500">
                          No accounts found matching the current filters.
                        </td>
                      </tr>
                    ) : (
                      filteredProfiles.map((profile) => (
                        <tr key={profile.id} className="hover:bg-gray-50 h-8">
                          <td className="px-3 py-1 whitespace-nowrap">
                            <div className="text-sm font-medium text-gray-900">
                              {profile.first_name && profile.last_name 
                                ? `${profile.first_name} ${profile.last_name}`
                                : profile.email.split('@')[0]}
                            </div>
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                            {profile.email}
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                            <RoleDisplay role={profile.role} />
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                            <StatusDisplay status={profile.status} />
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap text-gray-500 text-sm">
                            {format(new Date(profile.created_at), 'MMM d, yyyy')}
                          </td>
                          <td className="px-3 py-1 whitespace-nowrap text-right">
                            {profile.status === 'pending' && (
                              <div className="flex gap-2 justify-end">
                                <Button
                                  size="sm"
                                  onClick={() => handleApprove(profile.id)}
                                  className="bg-green-600 hover:bg-green-700 h-6 text-xs"
                                >
                                  <CheckCircle2 className="h-3 w-3 mr-1" />
                                  Approve
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleReject(profile.id)}
                                  className="text-red-600 border-red-200 hover:bg-red-50 h-6 text-xs"
                                >
                                  <XCircle className="h-3 w-3 mr-1" />
                                  Reject
                                </Button>
                              </div>
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
            </div>
          </div>
        ) : (
          /* For non-admin users, show a message */
          <Card>
            <CardHeader>
              <CardTitle>Account Access</CardTitle>
              <CardDescription>
                You can only view and manage your own account information.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Use the "My Account" button above to view and edit your profile information.
              </p>
            </CardContent>
          </Card>
        )}


        </div>
      </PageWrapper>
    </Layout>
  );
};

export default Accounts;