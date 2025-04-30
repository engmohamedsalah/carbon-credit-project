import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box } from '@mui/material';
import Dashboard from './pages/Dashboard';
import ProjectDetail from './pages/ProjectDetail';
import Verification from './pages/Verification';
import Login from './pages/Login';
import Register from './pages/Register';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import { useSelector } from 'react-redux';

function App() {
  const { isAuthenticated } = useSelector(state => state.auth);

  return (
    <Box sx={{ display: 'flex' }}>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/" element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="projects/:id" element={<ProjectDetail />} />
          <Route path="verification/:id" element={<Verification />} />
        </Route>
      </Routes>
    </Box>
  );
}

export default App;
