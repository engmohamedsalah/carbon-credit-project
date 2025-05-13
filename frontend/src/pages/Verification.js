import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Button,
  TextField,
  CircularProgress,
  Alert,
  Chip,
  Divider,
  Card,
  CardContent,
  CardActions,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useDispatch, useSelector } from 'react-redux';
import { useLocation, useNavigate } from 'react-router-dom';
import { verifyProject } from '../store/projectSlice';
import MapComponent from '../components/MapComponent';

const Verification = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const location = useLocation();
  
  // Get project_id from query params
  const query = new URLSearchParams(location.search);
  const projectId = query.get('project_id');
  
  const { verificationResults, loading, error } = useSelector(state => state.projects);
  const { user } = useSelector(state => state.auth);
  
  const [reviewNotes, setReviewNotes] = useState('');
  
  useEffect(() => {
    if (projectId) {
      dispatch(verifyProject(projectId));
    }
  }, [dispatch, projectId]);
  
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
  
  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4, textAlign: 'center' }}>
        <CircularProgress />
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
  
  if (!verificationResults) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="info">Verification not found</Alert>
      </Container>
    );
  }
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" gutterBottom>
          Project Verification
        </Typography>
        
        <Chip 
          label={verificationResults.status} 
          color={getStatusColor(verificationResults.status)}
          sx={{ textTransform: 'capitalize' }}
        />
      </Box>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Verification Results
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Project ID
                  </Typography>
                  <Typography variant="body1">
                    {verificationResults.project_id}
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} sm={6}>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Status
                  </Typography>
                  <Typography variant="body1">
                    {verificationResults.status}
                  </Typography>
                </Box>
              </Grid>
              
              {verificationResults.satellite_imagery && (
                <Grid item xs={12}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" color="text.primary">
                      Satellite Imagery
                    </Typography>
                    <Typography variant="body2">
                      Provider: {verificationResults.satellite_imagery.provider}
                    </Typography>
                    <Typography variant="body2">
                      Resolution: {verificationResults.satellite_imagery.resolution}
                    </Typography>
                    <Typography variant="body2">
                      Status: {verificationResults.satellite_imagery.status}
                    </Typography>
                  </Box>
                </Grid>
              )}
              
              {verificationResults.analysis_results && (
                <Grid item xs={12}>
                  <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Land Cover Analysis</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                        <Grid item xs={12}>
                          <Typography variant="body2" gutterBottom>
                            Total Area: {verificationResults.analysis_results.total_area_hectares} hectares
                        </Typography>
                          <Typography variant="body2" gutterBottom>
                            Total Carbon Estimate: {verificationResults.analysis_results.total_carbon_estimate_tonnes} tonnes
                        </Typography>
                          <Typography variant="body2" gutterBottom>
                            Confidence Level: {(verificationResults.analysis_results.confidence_level * 100).toFixed(1)}%
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12}>
                          <Typography variant="subtitle2" gutterBottom>
                            Land Cover Distribution
                        </Typography>
                          {verificationResults.analysis_results.land_cover_distribution && 
                            Object.entries(verificationResults.analysis_results.land_cover_distribution).map(([type, area]) => (
                              <Box key={type} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                <Typography variant="body2">{type}</Typography>
                                <Typography variant="body2">{area} ha</Typography>
                              </Box>
                            ))
                          }
                      </Grid>
                    </Grid>
                  </AccordionDetails>
                </Accordion>
                </Grid>
            )}
            </Grid>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
              Actions
                </Typography>
                
                  <Button 
                    variant="contained" 
              color="primary" 
              fullWidth 
              sx={{ mb: 2 }}
              onClick={() => navigate('/dashboard')}
            >
              Back to Dashboard
                  </Button>
                  
            {projectId !== 'new' && (
                  <Button 
                variant="outlined" 
                color="secondary" 
                    fullWidth
                onClick={() => navigate(`/projects/${projectId}`)}
                  >
                View Project Details
                  </Button>
              )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Verification;
