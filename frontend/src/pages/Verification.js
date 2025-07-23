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
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent
} from '@mui/material';
import { useSelector, useDispatch } from 'react-redux';
import { useLocation, useNavigate } from 'react-router-dom';
import { fetchProjects } from '../store/projectSlice';
import MLAnalysis from '../components/MLAnalysis';
import apiService from '../services/apiService';

const Verification = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useDispatch();
  
  // Get project_id from query params
  const query = new URLSearchParams(location.search);
  const urlProjectId = query.get('project_id');
  
  const { projects, loading, error } = useSelector(state => state.projects);
  // const { user } = useSelector(state => state.auth);
  const [selectedProjectId, setSelectedProjectId] = useState(urlProjectId || '');
  const [projectData, setProjectData] = useState(null);
  const [loadingProject, setLoadingProject] = useState(false);
  const [mlAnalysisResults, setMLAnalysisResults] = useState(null);
  
  useEffect(() => {
    dispatch(fetchProjects());
  }, [dispatch]);

  useEffect(() => {
    const fetchProjectData = async () => {
      setLoadingProject(true);
      try {
        const response = await apiService.get(`/projects/${selectedProjectId}`);
        setProjectData(response.data);
      } catch (error) {
        console.error('Failed to fetch project:', error);
        setProjectData(null);
      } finally {
        setLoadingProject(false);
      }
    };

    if (selectedProjectId) {
      fetchProjectData();
    } else {
      setProjectData(null);
    }
  }, [selectedProjectId]);

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
  
  if (!selectedProjectId && !loadingProject) {
    // Show project selection interface instead of error
  }
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Verification Hub
        </Typography>
        
        {projectData && (
        <Chip 
            label={projectData.status} 
            color={getStatusColor(projectData.status)}
          sx={{ textTransform: 'capitalize' }}
        />
        )}
      </Box>

      {/* Project Selection */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Select Project for Verification
          </Typography>
          <FormControl fullWidth>
            <InputLabel>Choose Project</InputLabel>
            <Select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              label="Choose Project"
            >
              <MenuItem value="">
                <em>Select a project...</em>
              </MenuItem>
              {Array.isArray(projects) && projects.map((project) => (
                <MenuItem key={project.id} value={project.id}>
                  {project.name} - {project.location_name} ({project.status})
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {!selectedProjectId && (
            <Alert severity="info" sx={{ mt: 2 }}>
              💡 <strong>Tip:</strong> This is the Verification Hub for bulk project management. 
              To verify individual projects, navigate to Projects → View Project → AI Verification tab.
            </Alert>
          )}
        </CardContent>
      </Card>
      
      {/* Project Information */}
      {selectedProjectId && projectData && (
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
      {selectedProjectId && projectData && (
        <MLAnalysis 
          projectId={parseInt(selectedProjectId)}
          projectData={projectData}
          onAnalysisComplete={handleAnalysisComplete}
        />
      )}

      {/* Analysis Summary (if completed) */}
      {selectedProjectId && mlAnalysisResults && (
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
                  
            {selectedProjectId && selectedProjectId !== 'new' && (
                  <Button 
            variant="contained" 
            color="primary"
                onClick={() => navigate(`/projects/${selectedProjectId}`)}
                  >
                View Project Details
                  </Button>
              )}
      </Box>
    </Container>
  );
};

export default Verification;
