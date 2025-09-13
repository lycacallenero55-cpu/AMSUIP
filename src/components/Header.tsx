import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from "@/components/ui/dropdown-menu";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { 
  GraduationCap, 
  User, 
  LogOut, 
  Menu,
  Eye,
  EyeOff,
  Edit,
  Save,
  X,
  Shield
} from "lucide-react";
import { UserCircle } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { useSidebar } from "@/contexts/SidebarContext";
import { supabase } from "@/lib/supabase";
import { fetchUserRole } from "@/lib/getUserRole";
import { cn } from "@/lib/utils";

// Extend window object for mobile drawer state
declare global {
  interface Window {
    mobileDrawerState?: {
      isOpen: boolean;
      toggle: () => void;
      close: () => void;
    };
  }
}

// Persistent role cache to prevent refetching and flashing on refresh
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

interface HeaderProps {
  isMobile?: boolean;
}

const Header = ({ isMobile = false }: HeaderProps) => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const { isCollapsed, toggleSidebar } = useSidebar();
  const [userRole, setUserRole] = useState<string>(() => {
    return cachedUserRole || 'user';
  });
  const [userProfile, setUserProfile] = useState<any>(null);
  const [academicYear, setAcademicYear] = useState<{
    year: string;
    semester: string;
  } | null>(null);
  const isInitialMount = useRef(true);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isLogoutConfirmOpen, setIsLogoutConfirmOpen] = useState(false);
  const [profileMode, setProfileMode] = useState<'display' | 'edit' | 'password'>('display');
  const [profileForm, setProfileForm] = useState<{ full_name: string; email: string }>({ full_name: '', email: '' });
  const [passwordForm, setPasswordForm] = useState<{ currentPassword: string; newPassword: string; confirmPassword: string }>({ currentPassword: '', newPassword: '', confirmPassword: '' });
  const [showPasswords, setShowPasswords] = useState<{ current: boolean; new: boolean; confirm: boolean }>({ current: false, new: false, confirm: false });
  const [isUpdatingProfile, setIsUpdatingProfile] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  useEffect(() => {
    const fetchRole = async () => {
      // If we have cached role for the same user, don't refetch
      if (cachedUserRole && cachedUserId === user?.id) {
        setUserRole(cachedUserRole);
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
        const role = await fetchUserRole(user.id);
        setUserRole(role);
        cachedUserRole = role;
        cachedUserId = user.id;
        setCachedUserRole(role, user.id);
        await fetchProfileData();
      } catch (error) {
        console.error('Error fetching user role:', error);
        const defaultRole = 'user';
        setUserRole(defaultRole);
        cachedUserRole = defaultRole;
        cachedUserId = user?.id || null;
        if (user?.id) {
          setCachedUserRole(defaultRole, user.id);
        }
      }
    };

    // Only fetch role on initial mount or when user changes
    if (isInitialMount.current || cachedUserId !== user?.id) {
      fetchRole();
      isInitialMount.current = false;
    }
  }, [user?.id]);

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

  // Fetch current academic year
  useEffect(() => {
    const fetchAcademicYear = async () => {
      try {
        // Mock academic year data - in production, this would fetch from actual database
        setAcademicYear({
          year: '2024-2025',
          semester: 'First Semester'
        });
      } catch (error) {
        console.error('Error fetching academic year:', error);
      }
    };
    fetchAcademicYear();
  }, []);

  const getPanelLabel = () => {
    if (userRole === 'admin') return 'Admin Panel';
    if (userRole === 'ROTC admin') return 'ROTC Admin Panel';
    if (userRole === 'Instructor') return 'Instructor Panel';
    if (userRole === 'SSG officer') return 'SSG Officer Panel';
    if (userRole === 'ROTC officer') return 'ROTC Officer Panel';
    return 'User Panel';
  };

  const getUserDisplayName = () => {
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

  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleProfileClick = () => {
    // Prefill form from loaded profile
    const fullName = userProfile?.first_name && userProfile?.last_name 
      ? `${userProfile.first_name} ${userProfile.last_name}`
      : userProfile?.first_name || userProfile?.last_name || '';
    
    setProfileForm({
      full_name: fullName,
      email: userProfile?.email || user?.email || ''
    });
    setProfileMode('display');
    setIsProfileOpen(true);
  };

  const handleProfileSave = async () => {
    try {
      if (!user) return;
      setIsUpdatingProfile(true);
      
      // Split full name into first and last name
      const nameParts = profileForm.full_name.trim().split(' ');
      const firstName = nameParts[0] || '';
      const lastName = nameParts.slice(1).join(' ') || '';
      
      // Update either admin or users table depending on where the profile came from
      if (userRole === 'admin') {
        const { error } = await supabase
          .from('admin')
          .update({ first_name: firstName, last_name: lastName })
          .eq('id', user.id);
        if (error) throw error;
      } else {
        const { error } = await supabase
          .from('users')
          .update({ first_name: firstName, last_name: lastName })
          .eq('id', user.id);
        if (error) throw error;
      }
      setUserProfile((prev: any) => ({ ...prev, first_name: firstName, last_name: lastName, email: profileForm.email }));
      setProfileMode('display');
    } catch (e) {
      console.error('Failed to save profile:', e);
    } finally {
      setIsUpdatingProfile(false);
    }
  };

  const handlePasswordChange = async () => {
    try {
      if (!user) return;
      
      // Validation
      if (!passwordForm.currentPassword.trim()) {
        console.error('Please enter your current password');
        return;
      }
      if (!passwordForm.newPassword.trim()) {
        console.error('Please enter a new password');
        return;
      }
      if (passwordForm.newPassword !== passwordForm.confirmPassword) {
        console.error('New passwords do not match');
        return;
      }
      if (passwordForm.newPassword.length < 6) {
        console.error('New password must be at least 6 characters long');
        return;
      }
      if (passwordForm.currentPassword === passwordForm.newPassword) {
        console.error('New password must be different from current password');
        return;
      }

      setIsChangingPassword(true);
      
      // First, verify the current password by attempting to sign in
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: user.email!,
        password: passwordForm.currentPassword
      });

      if (signInError) {
        console.error('Current password is incorrect');
        return;
      }

      // If current password is correct, update to new password
      const { error } = await supabase.auth.updateUser({
        password: passwordForm.newPassword
      });

      if (error) throw error;

      // Reset password form
      setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
      setProfileMode('display');
    } catch (error) {
      console.error('Error changing password:', error);
    } finally {
      setIsChangingPassword(false);
    }
  };

  if (isMobile) {
    const handleMobileMenuToggle = () => {
      if (window.mobileDrawerState) {
        window.mobileDrawerState.toggle();
      }
    };

    return (
      <header className="sticky top-0 z-50 md:hidden bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-sidebar-border h-14">
        <div className="flex items-center justify-between px-4 h-14">
          <div className="flex items-center gap-3">
            <div 
              className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-105"
              onClick={handleMobileMenuToggle}
            >
              <GraduationCap className="w-5 h-5 text-primary-foreground" />
            </div>
            <h1 className="text-lg font-bold text-education-navy">AMSUIP</h1>
          </div>
          <Button variant="ghost" size="icon" onClick={handleMobileMenuToggle}>
            <Menu className="h-5 w-5" />
          </Button>
        </div>
      </header>
    );
  }

  return (
    <header className="sticky top-0 z-40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-sidebar-border h-14">
      <div className="flex items-center justify-between pr-6 pl-2 h-14">
        {/* Left side - Logo and toggle */}
        <div className="flex items-center gap-4">
          <div 
            className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-105"
            onClick={toggleSidebar}
          >
            <GraduationCap className="w-5 h-5 text-primary-foreground" />
          </div>
          <h1 className="text-lg font-bold text-education-navy">AMSUIP</h1>
        </div>

        {/* Right side - Academic first, then Panel label, then User dropdown */}
        <div className="flex items-center gap-4">
          {/* Academic Year and Semester */}
          {academicYear && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>Current A.Y.:</span>
              <span>{academicYear.year}</span>
              <span>{academicYear.semester}</span>
            </div>
          )}

          {/* Vertical Separator */}
          {academicYear && (
            <div className="h-4 w-px bg-gray-300"></div>
          )}

          {/* Panel Label */}
          <div className="text-sm font-medium text-muted-foreground">
            {getPanelLabel()}
          </div>

          {/* User Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="relative h-8 w-8 rounded-lg bg-gradient-primary p-0 flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-105">
                <UserCircle className="h-5 w-5 text-primary-foreground" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56" align="end" forceMount>
              <div className="flex items-center justify-start gap-2 p-2">
                <div className="flex flex-col space-y-1 leading-none">
                  <p className="font-medium">{getUserDisplayName()}</p>
                </div>
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleProfileClick}>
                <User className="mr-2 h-4 w-4" />
                <span>Profile</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setIsLogoutConfirmOpen(true)}>
                <LogOut className="mr-2 h-4 w-4" />
                <span>Log out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      {/* Profile Dialog */}
      <Dialog open={isProfileOpen} onOpenChange={setIsProfileOpen}>
        <DialogContent className="max-w-md w-full">
          <DialogHeader>
            <DialogTitle>Profile Information</DialogTitle>
          </DialogHeader>
          
          {/* Fixed height container to prevent layout shifts */}
          <div className="min-h-[400px] space-y-4">
            
            {/* Display Mode - Default */}
            {profileMode === 'display' && (
              <div className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <span className="text-sm font-medium text-gray-700">Full Name:</span>
                    <p className="text-sm text-gray-900 mt-1">{profileForm.full_name || 'Not set'}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-700">Email:</span>
                    <p className="text-sm text-gray-900 mt-1">{profileForm.email}</p>
                  </div>
                  <div>
                    <span className="text-sm font-medium text-gray-700">Role:</span>
                    <p className="text-sm text-gray-900 mt-1">{getPanelLabel()}</p>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <Button 
                    variant="outline" 
                    onClick={() => setProfileMode('edit')}
                    className="flex-1"
                  >
                    <Edit className="h-4 w-4 mr-2" />
                    Edit Profile
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => setProfileMode('password')}
                    className="flex-1"
                  >
                    <Shield className="h-4 w-4 mr-2" />
                    Change Password
                  </Button>
                </div>
              </div>
            )}

            {/* Edit Mode */}
            {profileMode === 'edit' && (
              <div className="space-y-4">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="full_name">Full Name</Label>
                    <Input 
                      id="full_name" 
                      value={profileForm.full_name} 
                      onChange={(e) => setProfileForm(p => ({ ...p, full_name: e.target.value }))} 
                      placeholder="Enter your full name"
                    />
                  </div>
                  <div>
                    <Label htmlFor="email">Email</Label>
                    <Input 
                      id="email" 
                      value={profileForm.email} 
                      disabled 
                      className="bg-gray-50"
                    />
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <Button 
                    variant="outline" 
                    onClick={() => setProfileMode('display')}
                    className="flex-1"
                  >
                    <X className="h-4 w-4 mr-2" />
                    Cancel
                  </Button>
                  <Button 
                    onClick={handleProfileSave} 
                    disabled={isUpdatingProfile}
                    className="flex-1"
                  >
                    <Save className="h-4 w-4 mr-2" />
                    {isUpdatingProfile ? 'Saving...' : 'Save Changes'}
                  </Button>
                </div>
              </div>
            )}

            {/* Password Change Mode */}
            {profileMode === 'password' && (
              <div className="space-y-4">
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="currentPassword">Current Password</Label>
                    <div className="relative">
                      <Input
                        id="currentPassword"
                        type={showPasswords.current ? "text" : "password"}
                        value={passwordForm.currentPassword}
                        onChange={(e) => setPasswordForm(p => ({ ...p, currentPassword: e.target.value }))}
                        placeholder="Enter your current password"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                        onClick={() => setShowPasswords(p => ({ ...p, current: !p.current }))}
                      >
                        {showPasswords.current ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="newPassword">New Password</Label>
                    <div className="relative">
                      <Input
                        id="newPassword"
                        type={showPasswords.new ? "text" : "password"}
                        value={passwordForm.newPassword}
                        onChange={(e) => setPasswordForm(p => ({ ...p, newPassword: e.target.value }))}
                        placeholder="Enter your new password"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                        onClick={() => setShowPasswords(p => ({ ...p, new: !p.new }))}
                      >
                        {showPasswords.new ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="confirmPassword">Confirm New Password</Label>
                    <div className="relative">
                      <Input
                        id="confirmPassword"
                        type={showPasswords.confirm ? "text" : "password"}
                        value={passwordForm.confirmPassword}
                        onChange={(e) => setPasswordForm(p => ({ ...p, confirmPassword: e.target.value }))}
                        placeholder="Confirm your new password"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                        onClick={() => setShowPasswords(p => ({ ...p, confirm: !p.confirm }))}
                      >
                        {showPasswords.confirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <Button 
                    variant="outline" 
                    onClick={() => setProfileMode('display')}
                    className="flex-1"
                  >
                    <X className="h-4 w-4 mr-2" />
                    Cancel
                  </Button>
                  <Button 
                    onClick={handlePasswordChange} 
                    disabled={isChangingPassword}
                    className="flex-1"
                  >
                    <Shield className="h-4 w-4 mr-2" />
                    {isChangingPassword ? 'Changing...' : 'Change Password'}
                  </Button>
                </div>
              </div>
            )}
            
          </div>
        </DialogContent>
      </Dialog>

      {/* Logout Confirmation Dialog */}
      <Dialog open={isLogoutConfirmOpen} onOpenChange={setIsLogoutConfirmOpen}>
        <DialogContent className="max-w-sm w-full">
          <DialogHeader>
            <DialogTitle>Confirm Logout</DialogTitle>
          </DialogHeader>
          <p>Are you sure you want to log out?</p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsLogoutConfirmOpen(false)}>Cancel</Button>
            <Button variant="destructive" onClick={handleLogout}>Log Out</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </header>
  );
};

export default Header;