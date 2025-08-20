/**
 * ScrollToTop component that automatically scrolls the window to the top
 * when the route changes. Uses useEffect and useLocation to detect
 * navigation events and reset scroll position.
 */
import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const ScrollToTop = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
};

export default ScrollToTop;