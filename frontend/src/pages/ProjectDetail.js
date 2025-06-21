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
  Divider,
  MenuItem,
  Select,
  FormControl,
  InputLabel
} from '@mui/material';
import { useDispatch, useSelector } from 'react-redux';
import { useParams, useNavigate } from 'react-router-dom';
import { fetchProjectById, updateProjectStatus } from '../store/projectSlice';
import { fetchVerifications } from '../store/verificationSlice';
import MapComponent from '../components/MapComponent';

const ProjectDetail = () => {
  const { id } = useParams();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  const [statusUpdateLoading, setStatusUpdateLoading] = useState(false);
  
  const { currentProject, loading: projectLoading, error: projectError } = useSelector(state => state.projects);
  const { verifications, loading: verificationsLoading } = useSelector(state => state.verifications);
  const { user } = useSelector(state => state.auth);
  
  // Status options for progression
  const statusOptions = [
    'Draft',
    'Pending',
    'In Progress', 
    'Under Review',
    'Verified',
    'Rejected'
  ];
  
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

  const handleStatusUpdate = async (newStatus) => {
    setStatusUpdateLoading(true);
    try {
      await dispatch(updateProjectStatus({ 
        projectId: parseInt(id), 
        status: newStatus 
      })).unwrap();
      
      // Refresh project data
      dispatch(fetchProjectById(id));
    } catch (error) {
      console.error('Failed to update status:', error);
      alert('Failed to update project status. Please try again.');
    } finally {
      setStatusUpdateLoading(false);
    }
  };

  const canUpdateStatus = () => {
    // Project owner, Admins, and Verifiers can update status
    return currentProject?.user_id === user?.id || 
           user?.role === 'Admin' || 
           user?.role === 'Verifier';
  };

  // Helper function to render project field only if it has meaningful data
  const renderProjectField = (label, value, unit = '') => {
    if (!value && value !== 0) return null;
    
    let displayValue = value;
    if (label.includes('Carbon Credits') && typeof value === 'number') {
      displayValue = `${value} tonnes CO₂e`;
    } else if (label.includes('Hectares') && typeof value === 'number') {
      displayValue = `${value} hectares`;
    } else if (unit) {
      displayValue = `${value} ${unit}`;
    }
    
    return (
      <Box sx={{ mb: 1.5 }}>
        <Typography variant="body2" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="body1">
          {displayValue}
        </Typography>
      </Box>
    );
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
    <Container maxWidth={false} sx={{ mt: 4, mb: 4, px: 3, maxWidth: '90%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          {currentProject.name}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {canUpdateStatus() && (
            <FormControl size="small" sx={{ minWidth: 140 }}>
              <InputLabel>Update Status</InputLabel>
              <Select
                value={currentProject.status || 'Pending'}
                label="Update Status"
                onChange={(e) => handleStatusUpdate(e.target.value)}
                disabled={statusUpdateLoading}
              >
                {statusOptions.map((status) => (
                  <MenuItem key={status} value={status}>
                    {status}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          <Chip 
            label={currentProject.status || 'PENDING'} 
            color={currentProject.status === 'Verified' ? 'success' : 
                   currentProject.status === 'Rejected' ? 'error' :
                   currentProject.status === 'In Progress' ? 'primary' : 'default'}
            variant="outlined"
          />
        </Box>
      </Box>

      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ 
          display: 'flex', 
          gap: 4,
          flexDirection: { xs: 'column', md: 'row' },
          width: '100%',
          overflow: 'hidden'
        }}>
          {/* Project Details - Flexible content width */}
          <Box sx={{ 
            minWidth: 0,
            flex: { xs: '1', md: '1 1 60%' }
          }}>
            <Typography variant="h6" gutterBottom>
              Project Details
            </Typography>
            
            <Box sx={{ mb: 1.5 }}>
              <Typography variant="body2" color="text.secondary">
                Location
              </Typography>
              <Typography variant="body1">
                {currentProject.location_name || 'Not specified'}
              </Typography>
            </Box>
            
            <Box sx={{ mb: 1.5 }}>
              <Typography variant="body2" color="text.secondary">
                Project Type
              </Typography>
              <Typography variant="body1">
                {currentProject.project_type ? 
                  currentProject.project_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) 
                  : 'Not specified'
                }
              </Typography>
            </Box>
            
            {renderProjectField('Area (Hectares)', currentProject.area_hectares)}
            {renderProjectField('Start Date', formatDate(currentProject.start_date))}
            {renderProjectField('End Date', formatDate(currentProject.end_date))}
            {renderProjectField('Estimated Carbon Credits', currentProject.estimated_carbon_credits)}
            
            <Box sx={{ mb: 1.5 }}>
              <Typography variant="body2" color="text.secondary">
                Description
              </Typography>
              <Typography variant="body1">
                {currentProject.description || 'Not specified'}
              </Typography>
            </Box>
          </Box>
          
          {/* Map - Contained within frame */}
          {currentProject.geometry && (
            <Box sx={{ 
              minWidth: 0,
              flex: { xs: '1', md: '0 1 40%' },
              maxWidth: { md: '40%' }
            }}>
              <Typography variant="h6" gutterBottom>
                Project Area
              </Typography>
              <Box sx={{ 
                height: 350, 
                border: '1px solid #e0e0e0', 
                borderRadius: 1,
                width: '100%',
                overflow: 'hidden'
              }}>
                <MapComponent initialGeometry={currentProject.geometry} />
              </Box>
            </Box>
          )}
        </Box>
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
            ) : !Array.isArray(verifications) || verifications.length === 0 ? (
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
                          Carbon Impact
                        </Typography>
                        <Typography variant="body1">
                          {verification.carbon_impact ? `${verification.carbon_impact} tonnes CO₂/year` : 'Not specified'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          AI Confidence
                        </Typography>
                        <Typography variant="body1">
                          {verification.ai_confidence ? `${(verification.ai_confidence * 100).toFixed(1)}%` : 'Not specified'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          Certificate ID
                        </Typography>
                        <Typography variant="body1">
                          {verification.certificate_id || 'Not issued'}
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Typography variant="body2" color="text.secondary">
                          Human Verified
                        </Typography>
                        <Typography variant="body1">
                          {verification.human_verified ? 'Yes' : 'No'}
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
