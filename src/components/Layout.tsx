import Navigation from "@/components/ui/navigation";
import Header from "@/components/Header";
import { useMediaQuery } from "../hooks/use-media-query";
import { Skeleton } from "@/components/ui/skeleton";
import { useState, useEffect } from "react";
import { useSidebar } from "@/contexts/SidebarContext";
import { cn } from "@/lib/utils";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const isDesktop = useMediaQuery("(min-width: 768px)");
  const { isCollapsed } = useSidebar();
  const [isLoading, setIsLoading] = useState(true);



  useEffect(() => {
    // Prevent content flickering during route transitions
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 50);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header Component */}
      <Header isMobile={!isDesktop} />
      
      {/* Navigation Component (handles both desktop and mobile) */}
      <Navigation />
      
      {/* Main Content */}
      <main className={cn(
        "min-w-0",
        "transition-[margin-left] duration-250 ease-in-out", // Match navigation timing
        isDesktop 
          // Collapsed thinner, expanded restored to original
          ? (isCollapsed ? 'ml-12' : 'ml-64') 
          : 'ml-0 w-full px-4', // Add padding for mobile
        "pt-2 pb-3 md:pt-2 md:pb-4"
      )}>
        {isLoading ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Skeleton className="h-8 w-64" />
              <Skeleton className="h-9 w-32" />
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <Skeleton className="h-64 w-full" />
              <Skeleton className="h-64 w-full" />
            </div>
          </div>
        ) : (
          <>{children}</>
        )}
      </main>
    </div>
  );
};

export default Layout;