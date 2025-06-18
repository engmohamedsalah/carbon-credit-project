import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Button,
  CircularProgress,
  Alert,
  Chip
} from '@mui/material';
import { useSelector } from 'react-redux';
import { useLocation, useNavigate } from 'react-router-dom';
import MLAnalysis from '../components/MLAnalysis';
import apiService from '../services/apiService';

const Verification = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Get project_id from query params
  const query = new URLSearchParams(location.search);
  const projectId = query.get('project_id');
  
  const { loading, error } = useSelector(state => state.projects);
  const [projectData, setProjectData] = useState(null);
  const [loadingProject, setLoadingProject] = useState(false);
  const [mlAnalysisResults, setMLAnalysisResults] = useState(null);
  
  useEffect(() => {
    const fetchProjectData = async () => {
      setLoadingProject(true);
      try {
        const response = await apiService.get(`/projects/${projectId}`);
        setProjectData(response.data);
      } catch (error) {
        console.error('Failed to fetch project:', error);
      } finally {
        setLoadingProject(false);
      }
    };

    if (projectId) {
      fetchProjectData();
    }
  }, [projectId]);

  const handleAnalysisComplete = (results) => {
    setMLAnalysisResults(results);
    // You could also dispatch this to Redux store if needed
  };
  
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'pending':
        return 'default';
      case 'in progress':
        return 'primary';
      case 'verified':
        return 'success';
      case 'rejected':
        return 'error';
      default:
        return 'default';
    }
  };
  
  if (loading || loadingProject) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography variant="body1" sx={{ mt: 2 }}>
          {loadingProject ? 'Loading project data...' : 'Loading verification...'}
        </Typography>
      </Container>
    );
  }
  
  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }
  
  if (!projectData && !loadingProject) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="info">Project not found</Alert>
      </Container>
    );
  }
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          AI-Powered Carbon Credit Verification
        </Typography>
        
        {projectData && (
          <Chip 
            label={projectData.status} 
            color={getStatusColor(projectData.status)}
            sx={{ textTransform: 'capitalize' }}
          />
        )}
      </Box>
      
      {/* Project Information */}
      {projectData && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Project Information
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="text.secondary">Project Name</Typography>
              <Typography variant="body1">{projectData.name}</Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="text.secondary">Location</Typography>
              <Typography variant="body1">{projectData.location_name}</Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="text.secondary">Area Size</Typography>
              <Typography variant="body1">{projectData.area_size} hectares</Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="text.secondary">Project Type</Typography>
              <Typography variant="body1">{projectData.project_type}</Typography>
            </Grid>
          </Grid>
          {projectData.description && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">Description</Typography>
              <Typography variant="body1">{projectData.description}</Typography>
            </Box>
          )}
        </Paper>
      )}

      {/* ML Analysis Component */}
      <MLAnalysis 
        projectId={parseInt(projectId)}
        projectData={projectData}
        onAnalysisComplete={handleAnalysisComplete}
      />

      {/* Analysis Summary (if completed) */}
      {mlAnalysisResults && (
        <Paper sx={{ p: 3, mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Verification Summary
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={8}>
              <Typography variant="body1" gutterBottom>
                <strong>Recommendation:</strong> {mlAnalysisResults.eligibility?.recommendation}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                This assessment is based on machine learning analysis of satellite imagery, 
                location data, and forest cover patterns. Final certification requires 
                additional field verification and regulatory review.
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" color="primary">
                  {mlAnalysisResults.eligibility?.percentage}%
                </Typography>
                <Typography variant="body2">Eligibility Score</Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Navigation Actions */}
      <Box sx={{ mt: 3, display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Button 
          variant="outlined" 
          onClick={() => navigate('/dashboard')}
        >
          Back to Dashboard
        </Button>
        
        {projectId !== 'new' && (
          <Button 
            variant="contained" 
            color="primary"
            onClick={() => navigate(`/projects/${projectId}`)}
          >
            View Project Details
          </Button>
        )}
      </Box>
    </Container>
  );
};

export default Verification;
