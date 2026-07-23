import React from 'react';
import { Navigate } from 'react-router-dom';
import { useSelector } from 'react-redux';

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useSelector(state => state.auth);

  // Auth is still being checked (e.g. getCurrentUser rehydrating on boot/refresh).
  // Wait instead of redirecting, so a returning user isn't bounced to /login.
  if (loading) {
    return null;
  }

  // Check resolved and the user is not authenticated -> redirect to login.
  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }
  
  // If authenticated, render children
  return children;
};

export default ProtectedRoute;
