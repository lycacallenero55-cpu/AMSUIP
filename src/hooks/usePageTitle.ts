import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const SYSTEM_NAME = 'AMSUIP';

// Route to title mapping
const routeTitles: Record<string, string> = {
  '/': 'Dashboard',
  '/attendance': 'Attendance',
  '/reports': 'Reports',
  '/students': 'Students',
  '/schedule': 'Sessions',
  '/take-attendance': 'Take Attendance',
  '/accounts': 'Accounts',
  '/excuse-application': 'Excuse Application',
  '/academic-year': 'Allowed Terms',
  '/login': 'Login',
  '/sessions': 'Session Students',
};

export const usePageTitle = () => {
  const location = useLocation();

  useEffect(() => {
    const pathname = location.pathname;
    
    // Handle nested routes like /take-attendance/:sessionId
    let title = 'Dashboard'; // Default title
    
    // Check for exact matches first
    if (routeTitles[pathname]) {
      title = routeTitles[pathname];
    } else {
      // Handle nested routes by checking if pathname starts with known routes
      for (const [route, routeTitle] of Object.entries(routeTitles)) {
        if (route !== '/' && pathname.startsWith(route)) {
          title = routeTitle;
          break;
        }
      }
    }

    // Handle 404 page
    if (pathname === '*' || pathname.includes('404')) {
      title = 'Page Not Found';
    }

    // Set the document title
    document.title = `${title} - ${SYSTEM_NAME}`;
  }, [location.pathname]);
};