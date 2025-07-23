import React, { useState, useEffect, useCallback } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Card,
  CardContent,
  Alert,
  Button,
  CircularProgress,
  Snackbar,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Divider
} from '@mui/material';
import { 
  Assignment as AssignmentIcon,
  PictureAsPdf as PictureAsPdfIcon,
  Description as DescriptionIcon,
  BarChart as BarChartIcon,
  Download as DownloadIcon,
  GetApp as GetAppIcon
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import apiService from '../services/apiService';
import { COMMON_STYLES } from '../theme/constants';

const Reports = () => {
  const { user } = useSelector(state => state.auth);
  const [projects, setProjects] = useState([]);
  const [verifications, setVerifications] = useState([]);
  const [selectedProject, setSelectedProject] = useState('');
  const [loading, setLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  const fetchProjectsAndVerifications = useCallback(async () => {
    try {
      const [projectsResponse, verificationsResponse] = await Promise.all([
        apiService.get('/projects'),
        apiService.get('/verification')
      ]);
      
      // Ensure we have arrays, even if API returns different structure
      const projectsData = Array.isArray(projectsResponse.data) ? projectsResponse.data : 
                          (projectsResponse.data?.projects || projectsResponse.data?.data || []);
      const verificationsData = Array.isArray(verificationsResponse.data) ? verificationsResponse.data : 
                               (verificationsResponse.data?.verifications || verificationsResponse.data?.data || []);
      

      
      setProjects(projectsData);
      setVerifications(verificationsData);
    } catch (error) {
      console.error('Failed to fetch data:', error);
      showSnackbar('Failed to load data', 'error');
      // Set empty arrays as fallback
      setProjects([]);
      setVerifications([]);
    }
  }, []);

  useEffect(() => {
    fetchProjectsAndVerifications();
  }, [fetchProjectsAndVerifications]);

  const showSnackbar = (message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const downloadReport = async (endpoint, filename) => {
    setLoading(true);
    try {
      const response = await fetch(`${apiService.defaults.baseURL}${endpoint}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to generate report');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      showSnackbar('Report downloaded successfully', 'success');
    } catch (error) {
      console.error('Download failed:', error);
      showSnackbar('Failed to download report', 'error');
    } finally {
      setLoading(false);
    }
  };

  const downloadCertificate = (verificationId) => {
    downloadReport(
      `/reports/certificate/${verificationId}`,
      `verification_certificate_${verificationId}.pdf`
    );
  };

  const downloadProjectReport = (projectId) => {
    downloadReport(
      `/reports/project/${projectId}`,
      `project_report_${projectId}.pdf`
    );
  };

  const downloadAnalyticsReport = () => {
    downloadReport(
      '/reports/analytics',
      `analytics_report_${new Date().toISOString().split('T')[0]}.pdf`
    );
  };

  return (
    <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <AssignmentIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Typography variant="h4" gutterBottom>
          Reports & Documentation
        </Typography>
      </Box>

      <Alert severity="success" sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Report Generation Available:</strong> Download professional PDF reports, 
          verification certificates with QR codes, and comprehensive analytics summaries.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {/* Analytics Report */}
        {user?.role?.toLowerCase() === 'admin' && (
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <BarChartIcon sx={{ color: 'primary.main', mr: 2 }} />
                  <Typography variant="h6">
                    Analytics Report
                  </Typography>
                </Box>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  Comprehensive system analytics including performance metrics, 
                  verification trends, and carbon impact analysis.
                </Typography>
                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
                  onClick={downloadAnalyticsReport}
                  disabled={loading}
                  fullWidth
                >
                  Download Analytics Report
                </Button>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Project Reports */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <DescriptionIcon sx={{ color: 'secondary.main', mr: 2 }} />
                <Typography variant="h6">
                  Project Reports
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Detailed reports for individual projects including verification 
                history and carbon impact data.
              </Typography>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Select Project</InputLabel>
                <Select
                  value={selectedProject}
                  onChange={(e) => setSelectedProject(e.target.value)}
                  label="Select Project"
                >
                  {(projects || []).map(project => (
                    <MenuItem key={project.id} value={project.id}>
                      {project.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={20} /> : <DownloadIcon />}
                onClick={() => downloadProjectReport(selectedProject)}
                disabled={loading || !selectedProject}
                fullWidth
              >
                Download Project Report
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Verification Certificates */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <PictureAsPdfIcon sx={{ color: 'success.main', mr: 2 }} />
                <Typography variant="h6">
                  Verification Certificates
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Professional PDF certificates for verified carbon credit projects 
                with QR codes for authenticity verification.
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {(verifications || []).length} certificates available
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Verification Certificates List */}
      <Paper sx={{ mt: 3 }}>
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Available Verification Certificates
          </Typography>
        </Box>
        <Divider />
        
        {(verifications || []).length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No verification certificates available yet.
            </Typography>
          </Box>
        ) : (
          <List>
            {(verifications || []).map(verification => {
              const project = (projects || []).find(p => p.id === verification.project_id);
              return (
                <ListItem key={verification.id}>
                  <ListItemText
                    primary={project ? project.name : `Project #${verification.project_id}`}
                    secondary={
                      <Box>
                        <Typography variant="body2" component="span">
                          Status: {verification.status} • 
                          Carbon Impact: {verification.carbon_impact || 'N/A'} tonnes CO₂ • 
                          Date: {new Date(verification.created_at).toLocaleDateString()}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      onClick={() => downloadCertificate(verification.id)}
                      disabled={loading}
                      title="Download Certificate"
                    >
                      {loading ? <CircularProgress size={20} /> : <GetAppIcon />}
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              );
            })}
          </List>
        )}
      </Paper>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        message={snackbar.message}
      />
    </Container>
  );
};

export default Reports; 