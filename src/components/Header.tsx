import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { 
  GraduationCap, 
  User, 
  LogOut, 
  CalendarDays,
  Menu
} from "lucide-react";
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
    navigate('/profile');
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
      <div className="flex items-center justify-between px-6 h-14">
        {/* Left side - Logo and toggle */}
        <div className="flex items-center gap-4">
          <div 
            className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-105 -ml-1.5"
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
              <CalendarDays className="w-4 h-4" />
              <span>{academicYear.year}</span>
              <span>•</span>
              <span>{academicYear.semester}</span>
            </div>
          )}

          {/* Panel Label */}
          <div className="text-sm font-medium text-muted-foreground">
            {getPanelLabel()}
          </div>

          {/* User Dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                <Avatar className="h-8 w-8">
                  <AvatarFallback className="bg-gradient-primary text-primary-foreground text-sm">
                    {getUserDisplayName().split(' ').map(n => n[0]).join('').toUpperCase()}
                  </AvatarFallback>
                </Avatar>
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
              <DropdownMenuItem onClick={handleLogout}>
                <LogOut className="mr-2 h-4 w-4" />
                <span>Log out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
};

export default Header;