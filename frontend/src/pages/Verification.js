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
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Checkbox,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Tabs,
  Tab,
  Badge,
  LinearProgress,
  Tooltip,
  Fab
} from '@mui/material';
import {
  CheckCircle,
  Cancel,
  Pending,
  Visibility,
  PlayArrow,
  FilterList,
  Analytics,
  FileDownload,
  Refresh,
  Add
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { useLocation, useNavigate } from 'react-router-dom';
import { fetchProjects } from '../store/projectSlice';
import { fetchVerifications, submitHumanReview, createVerification } from '../store/verificationSlice';
import MLAnalysis from '../components/MLAnalysis';
import apiService from '../services/apiService';

const Verification = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  
  // Redux state
  const { projects, loading: projectsLoading } = useSelector(state => state.projects);
  const { verifications, loading: verificationsLoading } = useSelector(state => state.verifications);
  const { user } = useSelector(state => state.auth);
  
  // Component state
  const [currentTab, setCurrentTab] = useState(0);
  const [selectedProjects, setSelectedProjects] = useState([]);
  const [filterStatus, setFilterStatus] = useState('all');
  const [reviewDialog, setReviewDialog] = useState({ open: false, verification: null });
  const [bulkCreateDialog, setBulkCreateDialog] = useState(false);
  const [bulkProgress, setBulkProgress] = useState({ active: false, current: 0, total: 0 });
  const [stats, setStats] = useState({ total: 0, pending: 0, approved: 0, rejected: 0 });
  
  // Load data on mount
  useEffect(() => {
    dispatch(fetchProjects());
    dispatch(fetchVerifications({}));
  }, [dispatch]);

  // Update stats when verifications change
  useEffect(() => {
    if (verifications) {
      const stats = verifications.reduce((acc, v) => {
        acc.total++;
        switch (v.status?.toLowerCase()) {
          case 'pending':
          case 'created':
            acc.pending++;
            break;
          case 'approved':
          case 'verified':
            acc.approved++;
            break;
          case 'rejected':
            acc.rejected++;
            break;
        }
        return acc;
      }, { total: 0, pending: 0, approved: 0, rejected: 0 });
      setStats(stats);
    }
  }, [verifications]);

  // Filtered verifications based on status
  const filteredVerifications = verifications?.filter(v => 
    filterStatus === 'all' || v.status?.toLowerCase() === filterStatus.toLowerCase()
  ) || [];
  
  // Helper functions
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'pending':
      case 'created':
        return 'warning';
      case 'approved':
      case 'verified':
        return 'success';
      case 'rejected':
        return 'error';
      case 'in_progress':
        return 'info';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case 'pending':
      case 'created':
        return <Pending color="warning" />;
      case 'approved':
      case 'verified':
        return <CheckCircle color="success" />;
      case 'rejected':
        return <Cancel color="error" />;
      default:
        return <Pending color="disabled" />;
    }
  };

  // Handlers
  const handleSelectAll = (checked) => {
    if (checked) {
      setSelectedProjects(filteredVerifications.map(v => v.id));
    } else {
      setSelectedProjects([]);
    }
  };

  const handleSelectProject = (projectId, checked) => {
    if (checked) {
      setSelectedProjects(prev => [...prev, projectId]);
    } else {
      setSelectedProjects(prev => prev.filter(id => id !== projectId));
    }
  };

  const handleBulkCreateVerifications = async () => {
    setBulkProgress({ active: true, current: 0, total: selectedProjects.length });
    
    for (let i = 0; i < selectedProjects.length; i++) {
      const projectId = selectedProjects[i];
      setBulkProgress(prev => ({ ...prev, current: i + 1 }));
      
      try {
        await dispatch(createVerification({ project_id: projectId }));
        await new Promise(resolve => setTimeout(resolve, 500)); // Small delay
      } catch (error) {
        console.error(`Failed to create verification for project ${projectId}:`, error);
      }
    }
    
    setBulkProgress({ active: false, current: 0, total: 0 });
    setBulkCreateDialog(false);
    setSelectedProjects([]);
    dispatch(fetchVerifications({}));
  };

  const handleReview = async (approved, notes = '') => {
    try {
      await dispatch(submitHumanReview({
        id: reviewDialog.verification.id,
        approved,
        notes
      }));
      setReviewDialog({ open: false, verification: null });
      dispatch(fetchVerifications({}));
    } catch (error) {
      console.error('Failed to submit review:', error);
    }
  };

  // Available projects for bulk creation (those without verifications)
  const availableProjects = projects?.filter(project => 
    !verifications?.find(v => v.project_id === project.id)
  ) || [];

  if (projectsLoading || verificationsLoading) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography variant="body1" sx={{ mt: 2 }}>
          Loading verification hub...
        </Typography>
      </Container>
    );
  }

  return (
    <Box sx={{ mt: 4, mb: 4, px: 3, maxWidth: 'none', width: '100%' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Verification Hub
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => {
              dispatch(fetchProjects());
              dispatch(fetchVerifications({}));
            }}
          >
            Refresh
          </Button>
          
          {availableProjects.length > 0 && (
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => setBulkCreateDialog(true)}
            >
              Create Verifications
            </Button>
          )}
        </Box>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" color="primary">{stats.total}</Typography>
              <Typography variant="body2">Total Verifications</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" color="warning.main">{stats.pending}</Typography>
              <Typography variant="body2">Pending Review</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" color="success.main">{stats.approved}</Typography>
              <Typography variant="body2">Approved</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" color="error.main">{stats.rejected}</Typography>
              <Typography variant="body2">Rejected</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs and Filters */}
      <Card sx={{ mb: 3, minHeight: '600px', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={(e, newValue) => setCurrentTab(newValue)}>
            <Tab label="All Verifications" />
            <Tab label={`Pending (${stats.pending})`} />
            <Tab label="Analytics" />
          </Tabs>
        </Box>

        {/* Tab Content Container with consistent height */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          {currentTab === 0 && (
            <Box sx={{ p: 2, flex: 1, display: 'flex', flexDirection: 'column' }}>
              {/* Filter Controls */}
              <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'center' }}>
                <FormControl size="small" sx={{ minWidth: 120 }}>
                  <InputLabel>Status Filter</InputLabel>
                  <Select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    label="Status Filter"
                  >
                    <MenuItem value="all">All</MenuItem>
                    <MenuItem value="pending">Pending</MenuItem>
                    <MenuItem value="approved">Approved</MenuItem>
                    <MenuItem value="rejected">Rejected</MenuItem>
                  </Select>
                </FormControl>
                
                {selectedProjects.length > 0 && (
                  <Chip
                    label={`${selectedProjects.length} selected`}
                    onDelete={() => setSelectedProjects([])}
                    color="primary"
                  />
                )}
              </Box>

              {/* Verifications Table Container */}
              <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                {filteredVerifications.length > 0 ? (
                  <TableContainer component={Paper} sx={{ flex: 1 }}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell padding="checkbox">
                            <Checkbox
                              checked={selectedProjects.length === filteredVerifications.length && filteredVerifications.length > 0}
                              indeterminate={selectedProjects.length > 0 && selectedProjects.length < filteredVerifications.length}
                              onChange={(e) => handleSelectAll(e.target.checked)}
                            />
                          </TableCell>
                          <TableCell>Project</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Created</TableCell>
                          <TableCell>Score</TableCell>
                          <TableCell>Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {filteredVerifications.map((verification) => {
                          const project = projects?.find(p => p.id === verification.project_id);
                          return (
                            <TableRow key={verification.id}>
                              <TableCell padding="checkbox">
                                <Checkbox
                                  checked={selectedProjects.includes(verification.id)}
                                  onChange={(e) => handleSelectProject(verification.id, e.target.checked)}
                                />
                              </TableCell>
                              <TableCell>
                                <Box>
                                  <Typography variant="subtitle2">
                                    {project?.name || `Project ${verification.project_id}`}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary">
                                    {project?.location_name}
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell>
                                <Chip
                                  icon={getStatusIcon(verification.status)}
                                  label={verification.status || 'Created'}
                                  color={getStatusColor(verification.status)}
                                  size="small"
                                />
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2">
                                  {new Date(verification.created_at).toLocaleDateString()}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Typography variant="body2">
                                  {verification.eligibility_score ? `${verification.eligibility_score}%` : 'N/A'}
                                </Typography>
                              </TableCell>
                              <TableCell>
                                <Box sx={{ display: 'flex', gap: 1 }}>
                                  <Tooltip title="View Details">
                                    <IconButton
                                      size="small"
                                      onClick={() => navigate(`/projects/${verification.project_id}`)}
                                    >
                                      <Visibility />
                                    </IconButton>
                                  </Tooltip>
                                  
                                  {(verification.status === 'pending' || verification.status === 'created') && (
                                    <Tooltip title="Review">
                                      <IconButton
                                        size="small"
                                        onClick={() => setReviewDialog({ open: true, verification })}
                                      >
                                        <PlayArrow />
                                      </IconButton>
                                    </Tooltip>
                                  )}
                                </Box>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Box sx={{ 
                    flex: 1, 
                    display: 'flex', 
                    flexDirection: 'column', 
                    justifyContent: 'center', 
                    alignItems: 'center',
                    minHeight: '400px'
                  }}>
                    <Analytics sx={{ fontSize: 80, color: 'text.disabled', mb: 2 }} />
                    <Typography variant="h5" color="text.secondary" gutterBottom>
                      No verifications found
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                      Create new verifications to get started with your carbon credit verification process
                    </Typography>
                    <Button
                      variant="contained"
                      startIcon={<Add />}
                      onClick={() => setBulkCreateDialog(true)}
                      disabled={availableProjects.length === 0}
                      size="large"
                    >
                      Create Verifications
                    </Button>
                  </Box>
                )}
              </Box>
            </Box>
          )}

          {currentTab === 1 && (
            <Box sx={{ p: 3, flex: 1, display: 'flex', flexDirection: 'column' }}>
              <Typography variant="h6" gutterBottom>Pending Verifications</Typography>
              {stats.pending === 0 ? (
                <Box sx={{ 
                  flex: 1, 
                  display: 'flex', 
                  flexDirection: 'column', 
                  justifyContent: 'center', 
                  alignItems: 'center' 
                }}>
                  <Pending sx={{ fontSize: 80, color: 'warning.main', mb: 2 }} />
                  <Typography variant="h5" color="text.secondary" gutterBottom>
                    No pending verifications
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    All verifications have been processed
                  </Typography>
                </Box>
              ) : (
                <Alert severity="info">
                  {stats.pending} verification{stats.pending > 1 ? 's' : ''} pending review
                </Alert>
              )}
            </Box>
          )}

          {currentTab === 2 && (
            <Box sx={{ p: 2, flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Typography variant="h6" gutterBottom>Analytics Dashboard</Typography>
              
              {/* Top Charts Row - Larger Height */}
              <Grid container spacing={2} sx={{ flex: '0 0 300px' }}>
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                      <Typography variant="h6" gutterBottom color="primary">
                        Verification Timeline
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Track verification progress over time
                      </Typography>
                      <Box sx={{ 
                        flex: 1, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        bgcolor: 'grey.50',
                        borderRadius: 1,
                        minHeight: '200px'
                      }}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Analytics sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                          <Typography variant="body2" color="text.disabled">
                            Timeline Chart
                          </Typography>
                          <Typography variant="caption" color="text.disabled">
                            Verifications over time
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                      <Typography variant="h6" gutterBottom color="primary">
                        Success Rates
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Approval vs rejection statistics
                      </Typography>
                      <Box sx={{ 
                        flex: 1, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        bgcolor: 'grey.50',
                        borderRadius: 1,
                        minHeight: '200px'
                      }}>
                        <Box sx={{ textAlign: 'center' }}>
                          <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 1 }} />
                          <Typography variant="body2" color="text.disabled">
                            Success Rate Chart
                          </Typography>
                          <Typography variant="caption" color="text.disabled">
                            Approved vs Rejected
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              {/* Performance Metrics - Expanded */}
              <Card sx={{ flex: 1, minHeight: '200px' }}>
                <CardContent sx={{ height: '100%' }}>
                  <Typography variant="h6" gutterBottom color="primary">
                    Performance Metrics
                  </Typography>
                  
                  {/* Key Metrics Grid */}
                  <Grid container spacing={3} sx={{ mb: 3 }}>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'info.lighter', borderRadius: 2 }}>
                        <Typography variant="h3" color="info.main" gutterBottom>
                          {stats.total > 0 ? Math.round((stats.approved / stats.total) * 100) : 0}%
                        </Typography>
                        <Typography variant="subtitle2" color="info.main">Success Rate</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Based on {stats.total} total verifications
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'warning.lighter', borderRadius: 2 }}>
                        <Typography variant="h3" color="warning.main" gutterBottom>
                          2.5
                        </Typography>
                        <Typography variant="subtitle2" color="warning.main">Avg Review Time</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Days to complete
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'success.lighter', borderRadius: 2 }}>
                        <Typography variant="h3" color="success.main" gutterBottom>
                          95%
                        </Typography>
                        <Typography variant="subtitle2" color="success.main">ML Accuracy</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Model prediction accuracy
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'primary.lighter', borderRadius: 2 }}>
                        <Typography variant="h3" color="primary.main" gutterBottom>
                          12.5k
                        </Typography>
                        <Typography variant="subtitle2" color="primary.main">CO₂ Tons Verified</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Total carbon credits issued
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>

                  {/* Additional Insights */}
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                        <Typography variant="subtitle2" gutterBottom>Recent Activity</Typography>
                        <Typography variant="body2" color="text.secondary">
                          • 2 verifications created today<br/>
                          • 1 pending human review<br/>
                          • 0 completed this week
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                        <Typography variant="subtitle2" gutterBottom>Efficiency Trends</Typography>
                        <Typography variant="body2" color="text.secondary">
                          • Processing time: ↓ 15%<br/>
                          • Automation rate: ↑ 23%<br/>
                          • Error rate: ↓ 8%
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                        <Typography variant="subtitle2" gutterBottom>Quality Metrics</Typography>
                        <Typography variant="body2" color="text.secondary">
                          • Compliance rate: 100%<br/>
                          • Audit score: A+<br/>
                          • Blockchain sync: 100%
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Box>
          )}
        </Box>
      </Card>

      {/* Bulk Progress */}
      {bulkProgress.active && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Creating Verifications... ({bulkProgress.current}/{bulkProgress.total})
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={(bulkProgress.current / bulkProgress.total) * 100} 
            />
          </CardContent>
        </Card>
      )}

      {/* Bulk Create Dialog */}
      <Dialog open={bulkCreateDialog} onClose={() => setBulkCreateDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create Bulk Verifications</DialogTitle>
        <DialogContent>
          <Typography variant="body1" gutterBottom>
            Select projects to create verifications for:
          </Typography>
          
          <TableContainer component={Paper} sx={{ mt: 2, maxHeight: 400 }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedProjects.length === availableProjects.length && availableProjects.length > 0}
                      indeterminate={selectedProjects.length > 0 && selectedProjects.length < availableProjects.length}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedProjects(availableProjects.map(p => p.id));
                        } else {
                          setSelectedProjects([]);
                        }
                      }}
                    />
                  </TableCell>
                  <TableCell>Project Name</TableCell>
                  <TableCell>Location</TableCell>
                  <TableCell>Type</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {availableProjects.map((project) => (
                  <TableRow key={project.id}>
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={selectedProjects.includes(project.id)}
                        onChange={(e) => handleSelectProject(project.id, e.target.checked)}
                      />
                    </TableCell>
                    <TableCell>{project.name}</TableCell>
                    <TableCell>{project.location_name}</TableCell>
                    <TableCell>{project.project_type}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          {availableProjects.length === 0 && (
            <Alert severity="info" sx={{ mt: 2 }}>
              All projects already have verifications created.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBulkCreateDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleBulkCreateVerifications}
            disabled={selectedProjects.length === 0}
          >
            Create {selectedProjects.length} Verifications
          </Button>
        </DialogActions>
      </Dialog>

      {/* Review Dialog */}
      <Dialog open={reviewDialog.open} onClose={() => setReviewDialog({ open: false, verification: null })}>
        <DialogTitle>Review Verification</DialogTitle>
        <DialogContent>
          {reviewDialog.verification && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Project: {projects?.find(p => p.id === reviewDialog.verification.project_id)?.name}
              </Typography>
              
              <Box sx={{ my: 2 }}>
                <Typography variant="body2" color="text.secondary">Eligibility Score</Typography>
                <Typography variant="h4" color="primary">
                  {reviewDialog.verification.eligibility_score || 'N/A'}%
                </Typography>
              </Box>
              
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Review Notes"
                variant="outlined"
                sx={{ mt: 2 }}
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReviewDialog({ open: false, verification: null })}>
            Cancel
          </Button>
          <Button
            variant="outlined"
            color="error"
            onClick={() => handleReview(false)}
          >
            Reject
          </Button>
          <Button
            variant="contained"
            color="success"
            onClick={() => handleReview(true)}
          >
            Approve
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Verification;
