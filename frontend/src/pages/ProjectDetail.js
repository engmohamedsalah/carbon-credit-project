import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Button,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Chip,
  Divider
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useNavigate } from 'react-router-dom';
import { fetchProjectById } from '../store/projectSlice';
import { fetchVerifications } from '../store/verificationSlice';
import MapComponent from '../components/MapComponent';

const ProjectDetail = () => {
  const { id } = useParams();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  
  const { currentProject, loading: projectLoading, error: projectError } = useSelector(state => state.projects);
  const { verifications, loading: verificationsLoading } = useSelector(state => state.verifications);
  const { user } = useSelector(state => state.auth);
  
  useEffect(() => {
    dispatch(fetchProjectById(id));
    dispatch(fetchVerifications({ projectId: id }));
  }, [dispatch, id]);
  
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'draft':
        return 'default';
      case 'active':
        return 'primary';
      case 'under_verification':
        return 'warning';
      case 'verified':
        return 'success';
      case 'rejected':
        return 'error';
      default:
        return 'default';
    }
  };
  
  const formatDate = (dateString) => {
    if (!dateString) return 'Not set';
    return new Date(dateString).toLocaleDateString();
  };
  
  if (projectLoading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, textAlign: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }
  
  if (projectError) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">{projectError}</Alert>
      </Container>
    );
  }
  
  if (!currentProject) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="info">Project not found</Alert>
      </Container>
    );
  }
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" gutterBottom>
          {currentProject.name}
        </Typography>
        
        <Chip 
          label={currentProject.status.replace('_', ' ').toUpperCase()} 
          color={getStatusColor(currentProject.status)}
          sx={{ textTransform: 'capitalize' }}
        />
      </Box>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Project Details
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Location
              </Typography>
              <Typography variant="body1">
                {currentProject.location_name || 'Not specified'}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Project Type
              </Typography>
              <Typography variant="body1" sx={{ textTransform: 'capitalize' }}>
                {currentProject.project_type?.replace('_', ' ') || 'Not specified'}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Area
              </Typography>
              <Typography variant="body1">
                {currentProject.area_hectares ? `${currentProject.area_hectares} hectares` : 'Not specified'}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Date Range
              </Typography>
              <Typography variant="body1">
                {formatDate(currentProject.start_date)} to {formatDate(currentProject.end_date)}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Estimated Carbon Credits
              </Typography>
              <Typography variant="body1">
                {currentProject.estimated_carbon_credits ? `${currentProject.estimated_carbon_credits} tonnes CO₂e` : 'Not specified'}
              </Typography>
            </Box>
            
            {currentProject.blockchain_token_id && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Blockchain Token ID
                </Typography>
                <Typography variant="body1">
                  {currentProject.blockchain_token_id}
                </Typography>
              </Box>
            )}
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              Project Area
            </Typography>
            
            <Box sx={{ height: 300 }}>
              <MapComponent initialGeometry={currentProject.geometry} />
            </Box>
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Description
            </Typography>
            <Typography variant="body1">
              {currentProject.description || 'No description provided.'}
            </Typography>
          </Grid>
        </Grid>
      </Paper>
      
      <Paper sx={{ p: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange}>
            <Tab label="Verifications" />
            <Tab label="Satellite Images" />
            <Tab label="Carbon Estimates" />
          </Tabs>
        </Box>
        
        {/* Verifications Tab */}
        {tabValue === 0 && (
          <Box sx={{ pt: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Verification History
              </Typography>
              
              {(user?.role === 'verifier' || user?.role === 'admin') && (
                <Button 
                  variant="contained" 
                  color="primary"
                  onClick={() => navigate(`/verification/new?project=${id}`)}
                >
                  New Verification
                </Button>
              )}
            </Box>
            
            {verificationsLoading ? (
              <Box sx={{ textAlign: 'center', py: 3 }}>
                <CircularProgress />
              </Box>
            ) : verifications.length === 0 ? (
              <Alert severity="info">No verifications found for this project.</Alert>
            ) : (
              <Box>
                {verifications.map((verification) => (
                  <Paper key={verification.id} sx={{ p: 2, mb: 2 }}>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={6}>
                        <Typography variant="subtitle1">
                          Verification #{verification.id}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {formatDate(verification.verification_date || verification.created_at)}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6} sx={{ textAlign: 'right' }}>
                        <Chip 
                          label={verification.status.replace('_', ' ').toUpperCase()} 
                          color={getStatusColor(verification.status)}
                          sx={{ textTransform: 'capitalize' }}
                        />
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Divider sx={{ my: 1 }} />
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          Verified Carbon Credits
                        </Typography>
                        <Typography variant="body1">
                          {verification.verified_carbon_credits ? `${verification.verified_carbon_credits} tonnes CO₂e` : 'Not specified'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          Confidence Score
                        </Typography>
                        <Typography variant="body1">
                          {verification.confidence_score ? `${(verification.confidence_score * 100).toFixed(1)}%` : 'Not specified'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                        <Button 
                          variant="outlined" 
                          size="small"
                          onClick={() => navigate(`/verification/${verification.id}`)}
                        >
                          View Details
                        </Button>
                      </Grid>
                    </Grid>
                  </Paper>
                ))}
              </Box>
            )}
          </Box>
        )}
        
        {/* Satellite Images Tab */}
        {tabValue === 1 && (
          <Box sx={{ pt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Satellite Images
            </Typography>
            
            <Alert severity="info">
              Satellite image management will be implemented in the next phase.
            </Alert>
          </Box>
        )}
        
        {/* Carbon Estimates Tab */}
        {tabValue === 2 && (
          <Box sx={{ pt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Carbon Estimates
            </Typography>
            
            <Alert severity="info">
              Carbon estimate tracking will be implemented in the next phase.
            </Alert>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ProjectDetail;
