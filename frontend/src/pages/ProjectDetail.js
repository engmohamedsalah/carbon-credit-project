import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Grid,
  Button,
  Tabs,
  Tab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Alert,
  CircularProgress,
  Container,
  Paper
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  LocationOn as LocationIcon,
  CalendarToday as CalendarIcon,
  Eco as EcoIcon,
  Assessment as AssessmentIcon,
  History as HistoryIcon,
  CheckCircle as VerifiedIcon,
  Cancel as RejectedIcon,
  Pending as PendingIcon,
  Person as PersonIcon
} from '@mui/icons-material';
import { fetchProjectById, updateProjectStatus } from '../store/projectSlice';
import { fetchVerifications } from '../store/verificationSlice';
import MapComponent from '../components/MapComponent';
import apiService from '../services/apiService';

const ProjectDetail = () => {
  const { id } = useParams();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  const [statusUpdateLoading, setStatusUpdateLoading] = useState(false);
  const [statusDialogOpen, setStatusDialogOpen] = useState(false);
  const [selectedStatus, setSelectedStatus] = useState('');
  const [statusReason, setStatusReason] = useState('');
  const [statusNotes, setStatusNotes] = useState('');
  const [statusLogs, setStatusLogs] = useState([]);
  
  const { currentProject, loading: projectLoading, error: projectError } = useSelector(state => state.projects);
  const { verifications, loading: verificationsLoading } = useSelector(state => state.verifications);
  const { user } = useSelector(state => state.auth);
  
  // Status options for progression - simplified workflow
  const statusOptions = [
    'Draft',
    'Pending', 
    'Verified',
    'Rejected'
  ];
  
  const fetchStatusLogs = useCallback(async () => {
    try {
      const response = await apiService.get(`/api/v1/projects/${id}/status-logs`);
      setStatusLogs(response.data.status_logs || []);
    } catch (error) {
      console.error('Failed to fetch status logs:', error);
    }
  }, [id]);

  useEffect(() => {
    dispatch(fetchProjectById(id));
    dispatch(fetchVerifications({ projectId: id }));
    fetchStatusLogs();
  }, [dispatch, id, fetchStatusLogs]);
  
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

  const handleStatusChange = (newStatus) => {
    setSelectedStatus(newStatus);
    setStatusReason('');
    setStatusNotes('');
    setStatusDialogOpen(true);
  };

  const handleStatusUpdate = async () => {
    if (selectedStatus === 'Rejected' && !statusReason.trim()) {
      alert('Please provide a reason for rejection.');
      return;
    }

    setStatusUpdateLoading(true);
    try {
      await dispatch(updateProjectStatus({ 
        projectId: parseInt(id), 
        status: selectedStatus,
        reason: statusReason,
        notes: statusNotes
      })).unwrap();
      
      // Refresh project data and logs
      dispatch(fetchProjectById(id));
      fetchStatusLogs();
      setStatusDialogOpen(false);
    } catch (error) {
      console.error('Failed to update status:', error);
      alert(error.message || 'Failed to update project status. Please try again.');
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
                onChange={(e) => handleStatusChange(e.target.value)}
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
                   currentProject.status === 'Pending' ? 'warning' : 'default'}
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
            <Tab label="Status History" />
            <Tab label="Satellite Images" />
            <Tab label="Carbon Estimates" />
          </Tabs>
        </Box>
        
        {/* Status History Tab */}
        {tabValue === 1 && (
          <Box sx={{ pt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Status Change History
            </Typography>
            
            {statusLogs.length === 0 ? (
              <Alert severity="info">No status changes recorded yet.</Alert>
            ) : (
              <List>
                {statusLogs.map((log, index) => (
                                     <React.Fragment key={log.id}>
                     <ListItem>
                       <ListItemIcon>
                         {log.new_status === 'Verified' ? (
                           <VerifiedIcon color="success" />
                         ) : log.new_status === 'Rejected' ? (
                           <RejectedIcon color="error" />
                         ) : (
                           <PendingIcon color="warning" />
                         )}
                       </ListItemIcon>
                       <ListItemText
                         primary={
                           <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                             <Typography variant="h6" component="span">
                               {log.old_status ? `${log.old_status} → ${log.new_status}` : `Set to ${log.new_status}`}
                             </Typography>
                             <Typography variant="body2" color="text.secondary">
                               {new Date(log.created_at).toLocaleDateString()} {new Date(log.created_at).toLocaleTimeString()}
                             </Typography>
                           </Box>
                         }
                         secondary={
                           <Box>
                             <Typography variant="body2" color="text.secondary">
                               by {log.changed_by_name} ({log.changed_by_role})
                             </Typography>
                             {log.reason && (
                               <Box sx={{ mt: 1, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                                 <Typography variant="body2" fontWeight="bold">Reason:</Typography>
                                 <Typography variant="body2">{log.reason}</Typography>
                               </Box>
                             )}
                             {log.notes && (
                               <Box sx={{ mt: 1, p: 1, bgcolor: 'grey.50', borderRadius: 1 }}>
                                 <Typography variant="body2" fontWeight="bold">Notes:</Typography>
                                 <Typography variant="body2">{log.notes}</Typography>
                               </Box>
                             )}
                           </Box>
                         }
                       />
                     </ListItem>
                     {index < statusLogs.length - 1 && <Divider />}
                   </React.Fragment>
                ))}
              </List>
            )}
          </Box>
        )}

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
        {tabValue === 2 && (
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
        {tabValue === 3 && (
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

      {/* Status Change Dialog */}
      <Dialog 
        open={statusDialogOpen} 
        onClose={() => setStatusDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Update Project Status
        </DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <Typography variant="body1" gutterBottom>
              Change status from <strong>{currentProject?.status}</strong> to <strong>{selectedStatus}</strong>
            </Typography>
            
            {selectedStatus === 'Rejected' && (
              <TextField
                fullWidth
                required
                label="Reason for Rejection"
                value={statusReason}
                onChange={(e) => setStatusReason(e.target.value)}
                multiline
                rows={3}
                sx={{ mt: 2 }}
                helperText="Please provide a clear reason for rejecting this project"
              />
            )}
            
            {(selectedStatus === 'Verified' || selectedStatus === 'Rejected') && (
              <TextField
                fullWidth
                label={selectedStatus === 'Verified' ? 'Verification Notes' : 'Additional Notes'}
                value={statusNotes}
                onChange={(e) => setStatusNotes(e.target.value)}
                multiline
                rows={2}
                sx={{ mt: 2 }}
                helperText="Optional additional information about this decision"
              />
            )}

            {selectedStatus === 'Pending' && (
              <TextField
                fullWidth
                label="Submission Notes"
                value={statusNotes}
                onChange={(e) => setStatusNotes(e.target.value)}
                multiline
                rows={2}
                sx={{ mt: 2 }}
                helperText="Optional notes about this submission"
              />
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setStatusDialogOpen(false)}
            disabled={statusUpdateLoading}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleStatusUpdate}
            variant="contained"
            disabled={statusUpdateLoading || (selectedStatus === 'Rejected' && !statusReason.trim())}
          >
            {statusUpdateLoading ? <CircularProgress size={20} /> : 'Update Status'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ProjectDetail;
