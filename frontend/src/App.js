import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box } from '@mui/material';
import Dashboard from './pages/Dashboard';
import ProjectsList from './pages/ProjectsList';
import NewProject from './pages/NewProject';
import ProjectDetail from './pages/ProjectDetail';
import Verification from './pages/Verification';
import XAI from './pages/XAI';
import IoT from './pages/IoT';
import Analytics from './pages/Analytics';
import Blockchain from './pages/Blockchain';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import Login from './pages/Login';
import Register from './pages/Register';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
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
          <Route path="projects" element={<ProjectsList />} />
          <Route path="projects/new" element={<NewProject />} />
          <Route path="projects/:id" element={<ProjectDetail />} />
          <Route path="verification" element={<Verification />} />
          <Route path="verification/:id" element={<Verification />} />
          <Route path="xai" element={<XAI />} />
          <Route path="iot" element={<IoT />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="blockchain" element={<Blockchain />} />
          <Route path="reports" element={<Reports />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </Box>
  );
}

export default App;
