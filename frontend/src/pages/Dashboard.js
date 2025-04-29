import React from 'react';
import { Box, Typography, Container, Grid, Paper, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useSelector } from 'react-redux';

const Dashboard = () => {
  const navigate = useNavigate();
  const { user } = useSelector(state => state.auth);
  const { projects } = useSelector(state => state.projects);
  const { verifications } = useSelector(state => state.verifications);

  // Count projects by status
  const projectStats = {
    total: projects.length,
    active: projects.filter(p => p.status === 'active').length,
    verified: projects.filter(p => p.status === 'verified').length,
    underVerification: projects.filter(p => p.status === 'under_verification').length,
  };

  // Count verifications by status
  const verificationStats = {
    total: verifications.length,
    pending: verifications.filter(v => v.status === 'pending').length,
    humanReview: verifications.filter(v => v.status === 'human_review').length,
    approved: verifications.filter(v => v.status === 'approved').length,
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
                  Active: {projectStats.active}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Verified: {projectStats.verified}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Under Verification: {projectStats.underVerification}
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
        
        {/* Verification Stats */}
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
              Verifications
            </Typography>
            <Typography component="p" variant="h4">
              {verificationStats.total}
            </Typography>
            <Typography color="text.secondary" sx={{ flex: 1 }}>
              Total Verifications
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Pending: {verificationStats.pending}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Needs Human Review: {verificationStats.humanReview}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Approved: {verificationStats.approved}
                </Typography>
              </Box>
              {user?.role === 'verifier' || user?.role === 'admin' ? (
                <Button 
                  variant="contained" 
                  color="secondary"
                  onClick={() => navigate('/verifications')}
                >
                  Review Verifications
                </Button>
              ) : null}
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
              
              {user?.role === 'verifier' || user?.role === 'admin' ? (
                <Button 
                  variant="outlined" 
                  color="secondary"
                  onClick={() => navigate('/verification/new')}
                >
                  New Verification
                </Button>
              ) : null}
              
              <Button 
                variant="outlined"
                onClick={() => navigate('/blockchain/explorer')}
              >
                Blockchain Explorer
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
