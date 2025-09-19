import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { 
  LayoutDashboard, 
  UserCheck, 
  Users, 
  FileText, 
  CalendarClock,
  GraduationCap,
  Menu,
  X,
  UserCog,
  ClipboardCheck,
  School,
  Book,
  CalendarRange,
  LogOut,
  User,
  ChevronLeft,
  CalendarDays,
  BarChartBig,
  Brain
} from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useState, useEffect, useRef } from "react";
import { useMediaQuery } from "../../hooks/use-media-query";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/lib/supabase";
import { fetchUserRole, AppRole } from "@/lib/getUserRole";
import { useSidebar } from "@/contexts/SidebarContext";

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

const clearCachedUserRole = () => {
  try {
    localStorage.removeItem('userRole');
    localStorage.removeItem('userId');
  } catch {
    // Ignore localStorage errors
  }
};

let cachedUserRole: string | null = getCachedUserRole();
let cachedUserId: string | null = getCachedUserId();

// Navigation items configuration
const getNavItems = (userRole: string = '') => {
  if (userRole === 'admin') {
    // Admin - naturally full access
    return [
      { icon: LayoutDashboard, label: "Dashboard", href: "/" },
      { 
        icon: UserCheck, 
        label: "Take Attendance", 
        href: "/take-attendance",
        isActive: (path: string) => path === '/take-attendance' || path.startsWith('/take-attendance/')
      },
      { 
        icon: CalendarClock, 
        label: "Sessions", 
        href: "/schedule",
        isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
      },
      { icon: Users, label: "Students", href: "/students" },
      { icon: BarChartBig, label: "Reports", href: "/reports" },
      { icon: Book, label: "Subjects", href: "/subjects" },
      { 
        icon: FileText, 
        label: "Excuse Application", 
        href: "/excuse-application",
        isActive: (path: string) => path === '/excuse-application'
      },
      { icon: Brain, label: "Signature AI", href: "/signature-ai" },
      { icon: CalendarDays, label: "Allowed Terms", href: "/academic-year" },
      { icon: UserCog, label: "Accounts", href: "/accounts" }
    ];
  } else if (userRole === 'ROTC admin') {
    // ROTC Admin - same as admin but subject and allowed terms is restricted or hidden
    return [
      { icon: LayoutDashboard, label: "Dashboard", href: "/" },
      { 
        icon: UserCheck, 
        label: "Take Attendance", 
        href: "/take-attendance",
        isActive: (path: string) => path === '/take-attendance' || path.startsWith('/take-attendance/')
      },
      { 
        icon: CalendarClock, 
        label: "Sessions", 
        href: "/schedule",
        isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
      },
      { icon: Users, label: "Students", href: "/students" },
      { icon: BarChartBig, label: "Reports", href: "/reports" },
      { 
        icon: FileText, 
        label: "Excuse Application", 
        href: "/excuse-application",
        isActive: (path: string) => path === '/excuse-application'
      },
      { icon: Brain, label: "Signature AI", href: "/signature-ai" },
      { icon: UserCog, label: "Accounts", href: "/accounts" }
    ];
  } else if (userRole === 'Instructor') {
    // Instructor
    return [
      { icon: LayoutDashboard, label: "Dashboard", href: "/" },
      { 
        icon: UserCheck, 
        label: "Take Attendance", 
        href: "/take-attendance",
        isActive: (path: string) => path === '/take-attendance' || path.startsWith('/take-attendance/')
      },
      { 
        icon: CalendarClock, 
        label: "Sessions", 
        href: "/schedule",
        isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
      },
      { icon: Users, label: "Students", href: "/students" },
      { icon: BarChartBig, label: "Reports", href: "/reports" },
      { icon: Book, label: "Subjects", href: "/subjects" },
      { 
        icon: FileText, 
        label: "Excuse Application", 
        href: "/excuse-application",
        isActive: (path: string) => path === '/excuse-application'
      },
      { icon: Brain, label: "Signature AI", href: "/signature-ai" }
    ];
  } else if (userRole === 'SSG officer') {
    // SSG Officer - no subject
    return [
      { icon: LayoutDashboard, label: "Dashboard", href: "/" },
      { 
        icon: UserCheck, 
        label: "Take Attendance", 
        href: "/take-attendance",
        isActive: (path: string) => path === '/take-attendance' || path.startsWith('/take-attendance/')
      },
      { 
        icon: CalendarClock, 
        label: "Sessions", 
        href: "/schedule",
        isActive: (path: string) => path === '/schedule' || path.startsWith('/sessions/') 
      },
      { icon: Users, label: "Students", href: "/students" },
      { icon: BarChartBig, label: "Reports", href: "/reports" },
      { 
        icon: FileText, 
        label: "Excuse Application", 
        href: "/excuse-application",
        isActive: (path: string) => path === '/excuse-application'
      },
      { icon: Brain, label: "Signature AI", href: "/signature-ai" }
    ];
  } else if (userRole === 'ROTC officer') {
    // ROTC Officer - only Take Attendance, Profile, Log Out
    return [
      { 
        icon: UserCheck, 
        label: "Take Attendance", 
        href: "/take-attendance",
        isActive: (path: string) => path === '/take-attendance' || path.startsWith('/take-attendance/')
      }
    ];
  } else {
    // Default user role - limited access
    return [
      { icon: LayoutDashboard, label: "Dashboard", href: "/" },
      { 
        icon: UserCheck, 
        label: "Take Attendance", 
        href: "/take-attendance",
        isActive: (path: string) => path === '/take-attendance' || path.startsWith('/take-attendance/')
      },
      { 
        icon: FileText, 
        label: "Excuse Application", 
        href: "/excuse-application",
        isActive: (path: string) => path === '/excuse-application'
      }
    ];
  }
};

// Desktop Sidebar Navigation
const DesktopNavigation = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, signOut } = useAuth();
  const { isCollapsed, toggleSidebar } = useSidebar();
  const [userRole, setUserRole] = useState<string>(() => {
    // Initialize with cached role if available
    return cachedUserRole || 'user';
  });
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const isInitialMount = useRef(true);
    const navItems = getNavItems(userRole);
  
  // Debug logging
  console.log('Desktop Navigation - Current userRole:', userRole);
  console.log('Desktop Navigation - Generated navItems:', navItems);
  
  useEffect(() => {
    const fetchRole = async () => {
      // If we have cached role for the same user, don't refetch
      if (cachedUserRole && cachedUserId === user?.id) {
        setUserRole(cachedUserRole);
        return;
      }

      if (!user) {
        const defaultRole = 'user';
        setUserRole(defaultRole);
        cachedUserRole = defaultRole;
        cachedUserId = null;
        clearCachedUserRole();
        return;
      }
      
      try {
        const role = await fetchUserRole(user.id);
        setUserRole(role);
        cachedUserRole = role;
        cachedUserId = user.id;
        setCachedUserRole(role, user.id);
      } catch (error) {
        console.error('Error fetching user role:', error);
        const defaultRole = 'user';
        setUserRole(defaultRole);
        cachedUserRole = defaultRole;
        cachedUserId = user?.id || null;
        if (user?.id) {
          setCachedUserRole(defaultRole, user.id);
        } else {
          clearCachedUserRole();
        }
      }
    };

    // Only fetch role on initial mount or when user changes
    if (isInitialMount.current || cachedUserId !== user?.id) {
      fetchRole();
      isInitialMount.current = false;
    }
  }, [user?.id]); // Only depend on user ID, not the full user object

  const getPanelLabel = () => {
    if (userRole === 'admin') return 'Admin Panel';
    if (userRole === 'ROTC admin') return 'ROTC Admin Panel';
    if (userRole === 'Instructor') return 'Instructor Panel';
    if (userRole === 'SSG officer') return 'SSG Officer Panel';
    if (userRole === 'ROTC officer') return 'ROTC Officer Panel';
    return 'User Panel';
  };



  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleLogoutClick = () => {
    setShowLogoutConfirm(true);
  };

  const confirmLogout = async () => {
    setShowLogoutConfirm(false);
    await handleLogout();
  };

  const handleToggleSidebar = () => {
    // Width-only animation; items remain static
    toggleSidebar();
  };

  return (
    <TooltipProvider delayDuration={0}>
      <div className={cn(
        "h-full flex flex-col bg-background border-r border-sidebar-border",
        // Container-only width animation (fast & smooth); items remain static
        "transition-[width] duration-120 ease-in-out",
        isCollapsed ? "w-12" : "w-64"
      )} style={{height: 'calc(100vh - 56px)'}}>
      <div className={cn(
        "flex-1 px-2 pb-2 pt-2"
      )}>
        {/* Spacer removed to align items directly under header */}
        
        {/* Menu Items */}
        <div className={cn(
          "space-y-1.5 mb-0"
        )}>
          
          {navItems.map((item, index) => {
            
            const isActive = item.isActive 
              ? item.isActive(location.pathname) 
              : location.pathname === item.href;
            const Icon = item.icon;
            
            const menuItem = (
              <div
                className={cn(
                  "flex items-center cursor-pointer group relative",
                  // Only color/hover transitions; no positional animation
                  "transition-colors duration-200",
                  // Unified dimensions in both states to prevent distortion
                  "h-8 justify-start px-2 w-full rounded-sm overflow-hidden",
                  isActive 
                    ? "bg-gray-200 text-black font-semibold"
                    : "hover:bg-gray-100 hover:text-foreground font-normal"
                )}
              >
                <Icon className="flex-shrink-0 w-4 h-4" />
                <span className="ml-2 whitespace-nowrap min-w-0 flex-1 text-sm">
                  {item.label}
                </span>
              </div>
            );

            return (
              <Link key={item.href} to={item.href} className="block">
                {isCollapsed ? (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      {menuItem}
                    </TooltipTrigger>
                    <TooltipContent side="right" align="center" variant="sidebar">
                      {item.label}
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  menuItem
                )}
              </Link>
            );
          })}
        </div>

        {/* OTHER section removed; rely on header actions */}
      </div>
      
      {/* Logout Confirmation Dialog */}
      <Dialog open={showLogoutConfirm} onOpenChange={setShowLogoutConfirm}>
        <DialogContent className="!max-w-md w-[90vw] mx-auto rounded-lg">
          <DialogHeader>
            <DialogTitle>Confirm Logout</DialogTitle>
            <DialogDescription>
              Are you sure you want to log out? You will need to sign in again to access your account.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="!flex !flex-row !justify-end gap-2 sm:gap-3">
            <Button 
              variant="outline" 
              onClick={() => setShowLogoutConfirm(false)}
              className="flex-1 sm:flex-none sm:min-w-[80px]"
            >
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={confirmLogout}
              className="flex-1 sm:flex-none sm:min-w-[80px]"
            >
              Log Out
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      </div>
    </TooltipProvider>
  );
};

// Mobile Sidebar Navigation (Drawer)
const MobileDrawer = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, signOut } = useAuth();
  const [userRole, setUserRole] = useState<string>(() => {
    // Initialize with cached role if available
    return cachedUserRole || 'user';
  });
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const isInitialMount = useRef(true);
    const navItems = getNavItems(userRole);
  
  // Debug logging
  console.log('MobileDrawer - Current userRole:', userRole);
  console.log('MobileDrawer - Generated navItems:', navItems);
  
  // Handle animation states
  useEffect(() => {
    if (isOpen) {
      setIsVisible(true);
    } else {
      // Delay hiding to allow animation to complete
      const timer = setTimeout(() => {
        setIsVisible(false);
      }, 300); // Match transition duration
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  useEffect(() => {
    const fetchRole = async () => {
      // If we have cached role for the same user, don't refetch
      if (cachedUserRole && cachedUserId === user?.id) {
        setUserRole(cachedUserRole);
        return;
      }

      if (!user) {
        const defaultRole = 'user';
        setUserRole(defaultRole);
        cachedUserRole = defaultRole;
        cachedUserId = null;
        clearCachedUserRole();
        return;
      }
      
      try {
        const role = await fetchUserRole(user.id);
        setUserRole(role);
        cachedUserRole = role;
        cachedUserId = user.id;
        setCachedUserRole(role, user.id);
      } catch (error) {
        console.error('Error fetching user role:', error);
        const defaultRole = 'user';
        setUserRole(defaultRole);
        cachedUserRole = defaultRole;
        cachedUserId = user?.id || null;
        if (user?.id) {
          setCachedUserRole(defaultRole, user.id);
        } else {
          clearCachedUserRole();
        }
      }
    };

    // Only fetch role on initial mount or when user changes
    if (isInitialMount.current || cachedUserId !== user?.id) {
      fetchRole();
      isInitialMount.current = false;
    }
  }, [user?.id]); // Only depend on user ID, not the full user object

  const getPanelLabel = () => {
    if (userRole === 'admin') return 'Admin Panel';
    if (userRole === 'ROTC admin') return 'ROTC Admin Panel';
    if (userRole === 'Instructor') return 'Instructor Panel';
    if (userRole === 'SSG officer') return 'SSG Officer Panel';
    if (userRole === 'ROTC officer') return 'ROTC Officer Panel';
    return 'User Panel';
  };

  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleLogoutClick = () => {
    setShowLogoutConfirm(true);
  };

  const confirmLogout = async () => {
    setShowLogoutConfirm(false);
    await handleLogout();
  };
  
  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-[60] md:hidden">
      {/* Backdrop */}
      <div 
        className={cn(
          "fixed inset-0 bg-black/50 transition-opacity duration-300 ease-in-out",
          isOpen ? "opacity-100" : "opacity-0"
        )}
        onClick={onClose} 
      />
      
      {/* Drawer */}
      <div 
        className={cn(
          "fixed inset-y-0 left-0 w-64 bg-background p-6 overflow-y-auto",
          isOpen ? "animate-slide-in" : "animate-slide-out"
        )}
        style={{
          animationDuration: '300ms',
          animationTimingFunction: 'ease-in-out',
          animationFillMode: 'forwards'
        }}
      >
        <div className="flex justify-end items-center mb-8">
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-5 w-5" />
          </Button>
        </div>

        <div className="space-y-1.5">
          {navItems.map((item) => {
            const isActive = item.isActive 
              ? item.isActive(location.pathname) 
              : location.pathname === item.href;
            const Icon = item.icon;
            
            return (
              <Link key={item.href} to={item.href} onClick={onClose}>
                <Button
                  variant="ghost"
                  className={cn(
                    "w-full justify-start gap-3 h-10 text-sm transition-all duration-200 group relative overflow-hidden",
                    isActive 
                      ? "bg-gray-200 text-black font-semibold" 
                      : "hover:bg-gray-100 hover:text-foreground font-normal"
                  )}
                >
                  <Icon className="w-4.5 h-4.5" />
                  {item.label}
                </Button>
              </Link>
            );
          })}
        </div>

        {/* OTHER section removed; rely on header actions */}
      </div>

      {/* Logout Confirmation Dialog */}
      <Dialog open={showLogoutConfirm} onOpenChange={setShowLogoutConfirm}>
        <DialogContent className="!max-w-md w-[90vw] mx-auto rounded-lg">
          <DialogHeader>
            <DialogTitle>Confirm Logout</DialogTitle>
            <DialogDescription>
              Are you sure you want to log out? You will need to sign in again to access your account.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="!flex !flex-row !justify-end gap-2 sm:gap-3">
            <Button 
              variant="outline" 
              onClick={() => setShowLogoutConfirm(false)}
              className="flex-1 sm:flex-none sm:min-w-[80px]"
            >
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={confirmLogout}
              className="flex-1 sm:flex-none sm:min-w-[80px]"
            >
              Log Out
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

// Main Navigation Component
const Navigation = () => {
  const isDesktop = useMediaQuery("(min-width: 768px)");
  const [isMobileOpen, setIsMobileOpen] = useState(false);

  const toggleMobileDrawer = () => {
    setIsMobileOpen(!isMobileOpen);
  };

  const closeMobileDrawer = () => {
    setIsMobileOpen(false);
  };

  // Expose mobile drawer state to parent components
  useEffect(() => {
    // This allows the Header component to control mobile navigation
    window.mobileDrawerState = {
      isOpen: isMobileOpen,
      toggle: toggleMobileDrawer,
      close: closeMobileDrawer
    };
  }, [isMobileOpen]);



  if (isDesktop) {
    return (
      <div className="fixed left-0 top-14 h-[calc(100vh-56px)] z-40">
        <DesktopNavigation />
      </div>
    );
  }

  return (
    <>
      <MobileDrawer isOpen={isMobileOpen} onClose={closeMobileDrawer} />
    </>
  );
};

export default Navigation;