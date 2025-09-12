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
  Menu
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
  const [profileForm, setProfileForm] = useState<{ first_name: string; last_name: string; email: string }>({ first_name: '', last_name: '', email: '' });

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
    setProfileForm({
      first_name: userProfile?.first_name || '',
      last_name: userProfile?.last_name || '',
      email: userProfile?.email || user?.email || ''
    });
    setIsProfileOpen(true);
  };

  const handleProfileSave = async () => {
    try {
      if (!user) return;
      // Update either admin or users table depending on where the profile came from
      if (userRole === 'admin') {
        const { error } = await supabase
          .from('admin')
          .update({ first_name: profileForm.first_name, last_name: profileForm.last_name })
          .eq('id', user.id);
        if (error) throw error;
      } else {
        const { error } = await supabase
          .from('users')
          .update({ first_name: profileForm.first_name, last_name: profileForm.last_name })
          .eq('id', user.id);
        if (error) throw error;
      }
      setUserProfile((prev: any) => ({ ...prev, first_name: profileForm.first_name, last_name: profileForm.last_name, email: profileForm.email }));
      setIsProfileOpen(false);
    } catch (e) {
      console.error('Failed to save profile:', e);
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
              <span>•</span>
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
              <Button variant="ghost" className="relative h-8 w-8 rounded-full p-0 flex items-center justify-center">
                <UserCircle className="h-6 w-6" />
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
        <DialogContent className="max-w-lg w-full">
          <DialogHeader>
            <DialogTitle>Edit Profile</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-2">
              <Label htmlFor="first_name">First Name</Label>
              <Input id="first_name" value={profileForm.first_name} onChange={(e) => setProfileForm(p => ({ ...p, first_name: e.target.value }))} />
            </div>
            <div className="grid grid-cols-1 gap-2">
              <Label htmlFor="last_name">Last Name</Label>
              <Input id="last_name" value={profileForm.last_name} onChange={(e) => setProfileForm(p => ({ ...p, last_name: e.target.value }))} />
            </div>
            <div className="grid grid-cols-1 gap-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" value={profileForm.email} disabled />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsProfileOpen(false)}>Cancel</Button>
            <Button onClick={handleProfileSave}>Save</Button>
          </DialogFooter>
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