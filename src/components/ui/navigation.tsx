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
  const [isCollapsing, setIsCollapsing] = useState(false);
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
    if (!isCollapsed) {
      // Starting to collapse - instantly hide labels
      setIsCollapsing(true);
      // Start the actual collapse after a minimal delay to ensure instant fade
      setTimeout(() => {
        toggleSidebar();
        // Reset collapsing state after animation completes
        setTimeout(() => setIsCollapsing(false), 300);
      }, 50);
    } else {
      // Expanding - normal behavior
      toggleSidebar();
    }
  };

  return (
    <TooltipProvider delayDuration={0}>
      <div className={cn(
        "h-full flex flex-col bg-background border-r border-sidebar-border",
        // Stage 1: Width transition - horizontal shrinking/expanding
        "transition-[width] duration-250 ease-in-out",
        isCollapsed ? "w-12" : "w-64"
      )}>
      <div className={cn(
        "flex-1",
        isCollapsed ? "px-2 pb-2" : "p-4"
      )}>
        {/* Empty space where header used to be */}
        <div className="h-16"></div>
        
        {/* Menu Items */}
        <div className={cn(
          isCollapsed ? "space-y-2 mb-0 mt-0" : "space-y-1.5 mb-0"
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
                  // Stage 2: Item positioning - each with progressive delay
                  "transition-all duration-300 ease-in-out",
                  isCollapsed 
                    ? "h-8 justify-center w-6 mx-auto p-0 rounded-sm" // Larger container with smaller active background
                    : "h-9 justify-start gap-3 px-3 w-full rounded-sm", // Slightly smaller expanded items, perfect square-round
                  isActive 
                    ? isCollapsed 
                      ? "bg-gradient-primary shadow-glow text-white h-6 w-6 mx-auto rounded-sm"
                      : "bg-gradient-primary shadow-glow text-white"
                    : "hover:bg-sidebar-accent/50 hover:text-foreground"
                )}
                style={{
                  // Stage 2: Progressive delay for smooth upward movement
                  transitionDelay: isCollapsed ? `${250 + index * 30}ms` : '0ms'
                }}
              >
                  <Icon className={cn(
                    "flex-shrink-0",
                    // Icon sizes - smaller in open state, larger in collapsed state
                    isCollapsed ? "w-4 h-4" : "w-4 h-4" // Same size for consistency
                  )} />
                  
                  <span className={cn(
                    "font-medium whitespace-nowrap min-w-0",
                    // Stage 3: Text content - fades smoothly after positioning, instant fade when collapsing
                    isCollapsing ? "transition-opacity duration-75" : "transition-all duration-250 ease-in-out delay-300",
                    isCollapsed || isCollapsing
                      ? "opacity-0 translate-x-2 w-0 overflow-hidden text-xs" 
                      : "opacity-100 translate-x-0 flex-1 text-sm"
                  )}>
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
                    <TooltipContent side="right" align="center">
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

        {/* OTHER section - ZERO spacing above in any state */}
        <div className={cn(
          // ZERO spacing above OTHER in any state - should be IMMEDIATELY after menu items
          "mt-0",
          isCollapsed ? "space-y-1" : "space-y-1.5"
        )}>
          {/* OTHER Label */}
          {!isCollapsed && (
            <div className={cn(
              // Stage 2: Labels positioning - synchronized with menu items
              "transition-all duration-300 ease-in-out delay-200",
              "px-3 py-0"
            )}>
              <span className={cn(
                "font-medium text-muted-foreground/60 uppercase tracking-wider block",
                // Stage 3: Text styling - after positioning completes, instant fade when collapsing
                isCollapsing ? "transition-opacity duration-75" : "transition-all duration-300 ease-in-out delay-300",
                "text-xs opacity-100 text-left"
              )}>
                OTHER
              </span>
            </div>
          )}
          
          {/* Profile Item */}
          <Link to="/profile" className="block">
            {isCollapsed ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className={cn(
                      "flex items-center cursor-pointer group relative",
                      // Stage 2: Profile item positioning - after main menu items
                      "transition-all duration-300 ease-in-out delay-400",
                      isCollapsed 
                        ? "h-8 justify-center w-6 mx-auto p-0 rounded-sm" // Larger container with smaller active background
                        : "h-9 justify-start gap-3 px-3 w-full rounded-sm", // Slightly smaller expanded items, perfect square-round
                      location.pathname === '/profile'
                        ? isCollapsed 
                          ? "bg-gradient-primary shadow-glow text-white h-6 w-6 mx-auto rounded-sm"
                          : "bg-gradient-primary shadow-glow text-white"
                        : "hover:bg-sidebar-accent/50 hover:text-foreground"
                    )}
                  >
                    <User className={cn(
                      "flex-shrink-0",
                      // Icon sizes - consistent with main menu
                      isCollapsed ? "w-4 h-4" : "w-4 h-4"
                    )} />
                    <span className={cn(
                      "font-medium whitespace-nowrap min-w-0",
                      // Stage 3: Profile text - fades after positioning, instant fade when collapsing
                      isCollapsing ? "transition-opacity duration-75" : "transition-all duration-250 ease-in-out delay-450",
                      isCollapsed || isCollapsing
                        ? "opacity-0 translate-x-2 w-0 overflow-hidden text-xs" 
                        : "opacity-100 translate-x-0 flex-1 text-sm"
                    )}>
                      Profile
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="right" align="center">
                  Profile
                </TooltipContent>
              </Tooltip>
            ) : (
              <div
                className={cn(
                  "flex items-center cursor-pointer group relative",
                  // Stage 2: Profile item positioning - after main menu items
                  "transition-all duration-300 ease-in-out delay-400",
                  isCollapsed 
                    ? "h-8 justify-center w-6 mx-auto p-0 rounded-sm" // Larger container with smaller active background
                    : "h-9 justify-start gap-3 px-3 w-full rounded-sm", // Slightly smaller expanded items, perfect square-round
                  location.pathname === '/profile'
                    ? isCollapsed 
                      ? "bg-gradient-primary shadow-glow text-white h-6 w-6 mx-auto rounded-sm"
                      : "bg-gradient-primary shadow-glow text-white"
                    : "hover:bg-sidebar-accent/50 hover:text-foreground"
                )}
              >
                <User className={cn(
                  "flex-shrink-0",
                  // Icon sizes - consistent with main menu
                  isCollapsed ? "w-4 h-4" : "w-4 h-4"
                )} />
                <span className={cn(
                  "font-medium whitespace-nowrap min-w-0",
                  // Stage 3: Profile text - fades after positioning, instant fade when collapsing
                  isCollapsing ? "transition-opacity duration-75" : "transition-all duration-250 ease-in-out delay-450",
                  isCollapsed || isCollapsing
                    ? "opacity-0 translate-x-2 w-0 overflow-hidden text-xs" 
                    : "opacity-100 translate-x-0 flex-1 text-sm"
                )}>
                  Profile
                </span>
              </div>
            )}
          </Link>
          
          {/* Log Out Item */}
          {isCollapsed ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "flex items-center cursor-pointer group relative",
                    // Stage 2: Logout item positioning - final item
                    "transition-all duration-300 ease-in-out delay-450",
                    isCollapsed 
                      ? "h-8 justify-center w-6 mx-auto p-0 rounded-sm" // Larger container with smaller active background
                      : "h-9 justify-start gap-3 px-3 w-full rounded-sm", // Slightly smaller expanded items, perfect square-round
                    "hover:bg-destructive/10 hover:text-destructive text-destructive/90"
                  )}
                  onClick={handleLogoutClick}
                >
                  <LogOut className={cn(
                    "flex-shrink-0",
                    // Icon sizes - consistent with main menu
                    isCollapsed ? "w-4 h-4" : "w-4 h-4"
                  )} />
                  <span className={cn(
                    "font-medium whitespace-nowrap min-w-0",
                    // Stage 3: Logout text - fades after positioning, instant fade when collapsing
                    isCollapsing ? "transition-opacity duration-75" : "transition-all duration-250 ease-in-out delay-500",
                    isCollapsed || isCollapsing
                      ? "opacity-0 translate-x-2 w-0 overflow-hidden text-xs" 
                      : "opacity-100 translate-x-0 flex-1 text-sm"
                  )}>
                    Log Out
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent side="right" align="center">
                Log Out
              </TooltipContent>
            </Tooltip>
          ) : (
            <div
              className={cn(
                "flex items-center cursor-pointer group relative",
                // Stage 2: Logout item positioning - final item
                "transition-all duration-300 ease-in-out delay-450",
                isCollapsed 
                  ? "h-8 justify-center w-6 mx-auto p-0 rounded-sm" // Larger container with smaller active background
                  : "h-9 justify-start gap-3 px-3 w-full rounded-sm", // Slightly smaller expanded items, perfect square-round
                "hover:bg-destructive/10 hover:text-destructive text-destructive/90"
              )}
              onClick={handleLogoutClick}
            >
                              <LogOut className={cn(
                  "flex-shrink-0",
                  // Icon sizes - consistent with main menu
                  isCollapsed ? "w-4 h-4" : "w-4 h-4"
                )} />
              <span className={cn(
                "font-medium whitespace-nowrap min-w-0",
                // Stage 3: Logout text - fades after positioning, instant fade when collapsing
                isCollapsing ? "transition-opacity duration-75" : "transition-all duration-250 ease-in-out delay-500",
                isCollapsed || isCollapsing
                  ? "opacity-0 translate-x-2 w-0 overflow-hidden text-xs" 
                  : "opacity-100 translate-x-0 flex-1 text-sm"
              )}>
                Log Out
              </span>
            </div>
          )}
        </div>
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
                      ? "bg-gradient-primary shadow-glow text-white" 
                      : "hover:bg-[hsl(214,84%,56%)] hover:bg-opacity-10 hover:text-foreground"
                  )}
                >
                  <Icon className="w-4.5 h-4.5" />
                  {item.label}
                </Button>
              </Link>
            );
          })}
        </div>

        {/* OTHER section */}
        <div className="mt-auto space-y-1">
          <div className="px-3 py-1">
            <span className="text-xs font-medium text-muted-foreground/60 uppercase tracking-wider">
              OTHER
            </span>
          </div>
          
          <Link to="/profile" onClick={onClose}>
            <Button
              variant="ghost"
              className="w-full justify-start gap-3 h-10 text-sm transition-all duration-200 group relative overflow-hidden hover:bg-[hsl(214,84%,56%)] hover:bg-opacity-10 hover:text-foreground"
              style={{ margin: '1px 0' }}
            >
              <User className="w-4.5 h-4.5" />
              Profile
            </Button>
          </Link>
          
          <Button
            variant="ghost"
            className="w-full justify-start gap-3 h-10 text-sm transition-all duration-200 group relative overflow-hidden hover:bg-destructive/10 hover:text-destructive text-destructive/90"
            style={{ margin: '1px 0' }}
            onClick={handleLogoutClick}
          >
            <LogOut className="w-4.5 h-4.5" />
            Log Out
          </Button>
        </div>
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
      <div className="fixed left-0 top-0 h-full z-40">
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