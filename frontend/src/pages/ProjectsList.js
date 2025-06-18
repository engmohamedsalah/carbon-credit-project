import React, { useEffect, useState, useMemo, useCallback } from 'react';
import {
  Container,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Box,
  Chip,
  IconButton,
  Alert,
  CircularProgress,
  TextField,
  InputAdornment,
  Toolbar,
  Grid,
  Card
} from '@mui/material';
// Centralized utilities - eliminates DRY violations
import { getStatusColor } from '../utils/statusUtils';
import { formatDate as formatDateUtil } from '../utils/dateUtils';
import { COMMON_STYLES, DIMENSIONS, SPACING, COLORS } from '../theme/constants';
import {
  Add as AddIcon,
  Search as SearchIcon,
  Visibility as ViewIcon,
  VerifiedUser as VerifyIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { fetchProjects } from '../store/projectSlice';

const ProjectsList = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { projects, loading, error } = useSelector(state => state.projects);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    dispatch(fetchProjects());
  }, [dispatch]);

  // Use centralized utilities instead of local implementations
  // This eliminates code duplication and ensures consistency

  // Memoize expensive filtering operation
  const filteredProjects = useMemo(() => {
    if (!Array.isArray(projects)) return [];
    
    if (!searchTerm.trim()) return projects;
    
    const searchLower = searchTerm.toLowerCase();
    return projects.filter(project =>
      project.name?.toLowerCase().includes(searchLower) ||
      project.location_name?.toLowerCase().includes(searchLower) ||
      project.project_type?.toLowerCase().includes(searchLower)
    );
  }, [projects, searchTerm]);

  // Memoize navigation handlers
  const handleNewProject = useCallback(() => {
    navigate('/projects/new');
  }, [navigate]);

  const handleProjectView = useCallback((projectId) => {
    navigate(`/projects/${projectId}`);
  }, [navigate]);

  const handleProjectVerification = useCallback((projectId) => {
    navigate(`/verification?project_id=${projectId}`);
  }, [navigate]);

  if (loading) {
    return (
      <Container maxWidth="lg" sx={COMMON_STYLES.LOADING_CONTAINER}>
        <CircularProgress />
        <Typography variant="body1" sx={{ mt: SPACING.sm }}>
          Loading projects...
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ 
      mt: 0, 
      mb: 0, 
      pt: 1,
      pb: 0,
      minHeight: 'calc(100vh - 80px)',
      display: 'flex',
      flexDirection: 'column',
      width: '100%',
      maxWidth: '100% !important'
    }}>
      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ ...COMMON_STYLES.CARD, background: COLORS.gradients.primary, color: 'white' }}>
            <Typography variant="h4" fontWeight="bold">
              {filteredProjects.length}
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              Total Projects
            </Typography>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={COMMON_STYLES.CARD}>
            <Typography variant="h4" fontWeight="bold" color="success.main">
              {filteredProjects.filter(p => p.status === 'Verified').length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Verified
            </Typography>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" fontWeight="bold" color="warning.main">
              {filteredProjects.filter(p => p.status === 'Pending').length}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Pending Review
            </Typography>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="h4" fontWeight="bold" color="primary.main">
              {filteredProjects
                .reduce((sum, p) => sum + (p.area_hectares || 0), 0)
                .toLocaleString()}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Total Hectares
            </Typography>
          </Card>
        </Grid>
      </Grid>

      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h4" gutterBottom>
          My Projects
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={handleNewProject}
        >
          New Project
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Search and Filter Bar */}
      <Paper sx={{ p: 1.5, mb: 1 }}>
        <Toolbar sx={{ px: 0 }}>
          <TextField
            placeholder="Search projects..."
            variant="outlined"
            size="small"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ minWidth: DIMENSIONS.SEARCH_MIN_WIDTH }}
          />
          <Box sx={{ flexGrow: 1 }} />
          <Typography variant="body2" color="text.secondary">
            {filteredProjects.length} project{filteredProjects.length !== 1 ? 's' : ''}
          </Typography>
        </Toolbar>
      </Paper>

      {/* Projects Table */}
      <TableContainer 
        component={Paper} 
        sx={{ 
          flexGrow: 1, 
          display: 'flex', 
          flexDirection: 'column', 
          width: '100%',
          height: '100%',
          overflow: 'auto'
        }}
      >
        <Table sx={{ 
          height: '100%', 
          width: '100%',
          minHeight: 'calc(100vh - 300px)' 
        }}>
          <TableHead>
            <TableRow>
              <TableCell>Project Name</TableCell>
              <TableCell>Location</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Area (ha)</TableCell>
              <TableCell>Created</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody sx={{ height: '100%', width: '100%' }}>
            {filteredProjects.length === 0 ? (
              <TableRow sx={{ height: '100%', width: '100%' }}>
                <TableCell 
                  colSpan={7} 
                  align="center" 
                  sx={{ 
                    height: '100%',
                    width: '100%',
                    verticalAlign: 'middle',
                    border: 'none',
                    minHeight: '500px',
                    padding: 0
                  }}
                >
                  <Box 
                    className="full-width-empty-state"
                    sx={{ 
                      textAlign: 'center', 
                      width: '100% !important',
                      maxWidth: 'none !important',
                      mx: 0,
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      height: '100%',
                      minHeight: '500px',
                      px: 4,
                      '&.MuiBox-root': {
                        width: '100% !important',
                        maxWidth: 'none !important'
                      }
                    }}>
                    <Box sx={{ mb: 4, fontSize: '4rem', lineHeight: 1 }}>🌱</Box>
                    <Typography variant="h4" color="text.primary" gutterBottom fontWeight="medium" sx={{ mb: 2 }}>
                      {searchTerm ? 'No matching projects' : 'Start Your Carbon Journey'}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mb: 5, lineHeight: 1.7, fontSize: '1.1rem' }}>
                      {searchTerm 
                        ? 'Try adjusting your search terms or create a new project to begin your environmental impact tracking' 
                        : 'Create your first carbon credit project and begin making a measurable positive environmental impact. Track forest restoration, carbon sequestration, and verification progress.'}
                    </Typography>
                    <Box>
                      <Button
                        variant="contained"
                        color="primary"
                        size="large"
                        startIcon={<AddIcon />}
                        onClick={handleNewProject}
                        sx={{ 
                          px: 5, 
                          py: 2, 
                          fontSize: '1.1rem',
                          borderRadius: 2,
                          textTransform: 'none',
                          fontWeight: 600
                        }}
                      >
                        {searchTerm ? 'Create New Project' : 'Create First Project'}
                      </Button>
                    </Box>
                    
                    {/* Additional helpful info */}
                    {!searchTerm && (
                      <Box sx={{ mt: 6, pt: 4, borderTop: '1px solid', borderColor: 'divider' }}>
                        <Grid container spacing={3} sx={{ textAlign: 'center' }}>
                          <Grid item xs={12} md={4}>
                            <Box sx={{ mb: 1, fontSize: '1.5rem' }}>🌳</Box>
                            <Typography variant="subtitle2" fontWeight="medium">Forest Restoration</Typography>
                            <Typography variant="body2" color="text.secondary">Track reforestation projects</Typography>
                          </Grid>
                          <Grid item xs={12} md={4}>
                            <Box sx={{ mb: 1, fontSize: '1.5rem' }}>📊</Box>
                            <Typography variant="subtitle2" fontWeight="medium">ML Verification</Typography>
                            <Typography variant="body2" color="text.secondary">AI-powered analysis</Typography>
                          </Grid>
                          <Grid item xs={12} md={4}>
                            <Box sx={{ mb: 1, fontSize: '1.5rem' }}>🏆</Box>
                            <Typography variant="subtitle2" fontWeight="medium">Carbon Credits</Typography>
                            <Typography variant="body2" color="text.secondary">Generate verified credits</Typography>
                          </Grid>
                        </Grid>
                      </Box>
                    )}
                  </Box>
                </TableCell>
              </TableRow>
            ) : (
              filteredProjects.map((project) => (
                <TableRow key={project.id} hover>
                  <TableCell>
                    <Typography variant="subtitle2" fontWeight="medium">
                      {project.name}
                    </Typography>
                    {project.description && (
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                        {project.description.length > 60 
                          ? `${project.description.substring(0, 60)}...` 
                          : project.description
                        }
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {project.location_name || 'Not specified'}
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                      {project.project_type?.replace('_', ' ') || 'Not specified'}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={project.status || 'Draft'} 
                      color={getStatusColor(project.status)}
                      size="small"
                      sx={{ textTransform: 'capitalize' }}
                    />
                  </TableCell>
                  <TableCell>
                    {project.area_hectares ? `${project.area_hectares.toLocaleString()}` : 'Not set'}
                  </TableCell>
                  <TableCell>
                    {formatDateUtil(project.created_at)}
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <IconButton
                        size="small"
                        onClick={() => handleProjectView(project.id)}
                        title="View Details"
                      >
                        <ViewIcon />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => handleProjectVerification(project.id)}
                        title="Verify Project"
                        color="primary"
                      >
                        <VerifyIcon />
                      </IconButton>
                    </Box>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Additional insights when projects exist */}
      {filteredProjects.length > 5 && (
        <Paper sx={{ p: 2, mt: 1 }}>
          <Typography variant="subtitle1" gutterBottom>
            Project Insights
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Average Size
                </Typography>
                <Typography variant="h6">
                  {Math.round(filteredProjects.reduce((sum, p) => sum + (p.area_hectares || 0), 0) / filteredProjects.length)} ha
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Completion Rate
                </Typography>
                <Typography variant="h6" color="success.main">
                  {Math.round((filteredProjects.filter(p => p.status === 'Verified').length / filteredProjects.length) * 100)}%
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Total Area
                </Typography>
                <Typography variant="h6">
                  {filteredProjects.reduce((sum, p) => sum + (p.area_hectares || 0), 0).toLocaleString()} ha
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Container>
  );
};

export default ProjectsList; 