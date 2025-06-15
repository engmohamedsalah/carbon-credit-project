import React, { useEffect } from 'react';
import { Box, Typography, Container, Grid, Paper, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { fetchProjects } from '../store/projectSlice';

const Dashboard = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { user } = useSelector(state => state.auth);
  const { projects, loading } = useSelector(state => state.projects);

  useEffect(() => {
    dispatch(fetchProjects());
  }, [dispatch]);

  // Count projects by status - ensure projects is always an array
  const projectsArray = Array.isArray(projects) ? projects : [];
  const projectStats = {
    total: projectsArray.length,
    pending: projectsArray.filter(p => p.status === 'Pending').length,
    verified: projectsArray.filter(p => p.status === 'Verified').length,
    inProgress: projectsArray.filter(p => p.status === 'In Progress').length,
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Typography variant="h6" gutterBottom>
        Welcome, {user?.full_name || 'User'}
      </Typography>
      
      <Grid container spacing={3}>
        {/* Project Stats */}
        <Grid item xs={12} md={6}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 240,
            }}
          >
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Projects
            </Typography>
            <Typography component="p" variant="h4">
              {projectStats.total}
            </Typography>
            <Typography color="text.secondary" sx={{ flex: 1 }}>
              Total Projects
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Pending: {projectStats.pending}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Verified: {projectStats.verified}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  In Progress: {projectStats.inProgress}
                </Typography>
              </Box>
              <Button 
                variant="contained" 
                color="primary"
                onClick={() => navigate('/projects')}
              >
                View Projects
              </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Satellite Stats */}
        <Grid item xs={12} md={6}>
          <Paper
            sx={{
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              height: 240,
            }}
          >
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Satellite Imagery
            </Typography>
            <Typography component="p" variant="h4">
              {projectStats.verified}
            </Typography>
            <Typography color="text.secondary" sx={{ flex: 1 }}>
              Verified Projects
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Land Cover Analysis
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Carbon Estimation
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Sentinel-2 Imagery
                </Typography>
              </Box>
                <Button 
                  variant="contained" 
                  color="secondary"
                onClick={() => navigate('/verification?project_id=new')}
                >
                New Verification
                </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Quick Actions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography component="h2" variant="h6" color="primary" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button 
                variant="outlined" 
                color="primary"
                onClick={() => navigate('/projects/new')}
              >
                New Project
              </Button>
              
                <Button 
                  variant="outlined" 
                  color="secondary"
                onClick={() => navigate('/verification?project_id=new')}
                >
                Verify Project
                </Button>
              
              <Button 
                variant="outlined"
                onClick={() => navigate('/dashboard')}
              >
                View Reports
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
