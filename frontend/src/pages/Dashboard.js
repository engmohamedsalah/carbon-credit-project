import React, { useEffect, useState } from 'react';
import { Box, Typography, Container, Grid, Button, CircularProgress, Chip, IconButton } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { fetchProjects } from '../store/projectSlice';
import { GlassPanel, GlassCard, PipelineVisualization } from '../components/modern';
import { colors, spacing } from '../components/modern';

// Modern icons
import DashboardIcon from '@mui/icons-material/Dashboard';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import ForestIcon from '@mui/icons-material/Forest';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import VerifiedIcon from '@mui/icons-material/Verified';
import PendingActionsIcon from '@mui/icons-material/PendingActions';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import RefreshIcon from '@mui/icons-material/Refresh';

const Dashboard = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { user } = useSelector(state => state.auth);
  const { projects, loading } = useSelector(state => state.projects);
  const [currentPipelineStep, setCurrentPipelineStep] = useState(4);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    dispatch(fetchProjects());
    // Auto-refresh every 30 seconds for live data feel
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 30000);
    return () => clearInterval(interval);
  }, [dispatch]);

  // Count projects by status - ensure projects is always an array
  const projectsArray = Array.isArray(projects) ? projects : [];
  const projectStats = {
    total: projectsArray.length,
    pending: projectsArray.filter(p => p.status === 'Pending').length,
    verified: projectsArray.filter(p => p.status === 'Verified').length,
    inProgress: projectsArray.filter(p => p.status === 'In Progress').length,
  };

  // Enhanced stats for mission control
  const missionStats = {
    carbonCreditsIssued: 15420,
    satelliteImages: 1847,
    aiAnalyses: 892,
    iotSensors: 234,
    systemStatus: 'operational'
  };

  const handleRefresh = () => {
    dispatch(fetchProjects());
    setLastUpdate(new Date());
  };

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ 
        py: spacing.xl,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh'
      }}>
        <CircularProgress sx={{ color: colors.accent.primary, mb: spacing.md }} />
        <Typography variant="h6" sx={{ color: colors.text.primary, fontFamily: 'JetBrains Mono' }}>
          INITIALIZING MISSION CONTROL...
        </Typography>
        <Typography variant="body2" sx={{ color: colors.text.secondary, mt: spacing.sm }}>
          Loading satellite data and system status
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: spacing.sm, px: spacing.lg }}>
      {/* Mission Control Header */}
      <Box sx={{ mb: spacing.sm }}>
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          mb: spacing.md 
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: spacing.md }}>
            <DashboardIcon sx={{ 
              fontSize: '2.5rem', 
              color: colors.accent.primary,
              filter: `drop-shadow(0 0 10px ${colors.accent.primary}60)`
            }} />
            <Box>
              <Typography variant="h3" sx={{ 
                color: colors.text.primary,
                fontWeight: 700,
                letterSpacing: '-0.02em',
                textTransform: 'uppercase',
                fontFamily: 'Inter'
              }}>
                Mission Control
              </Typography>
              <Typography variant="h6" sx={{ 
                color: colors.text.secondary,
                fontWeight: 400,
                mt: -0.5
              }}>
                Carbon Credit Verification System
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: spacing.md }}>
            <Chip 
              label={missionStats.systemStatus.toUpperCase()}
              size="small"
              sx={{
                backgroundColor: colors.accent.primary,
                color: colors.background.primary,
                fontWeight: 600,
                fontSize: '0.75rem',
                animation: 'pulse 2s infinite'
              }}
            />
            <Typography variant="caption" sx={{ 
              color: colors.text.tertiary,
              fontFamily: 'JetBrains Mono'
            }}>
              Last Update: {lastUpdate.toLocaleTimeString()}
            </Typography>
            <IconButton
              onClick={handleRefresh}
              sx={{ 
                color: colors.accent.secondary,
                '&:hover': { 
                  color: colors.accent.primary,
                  backgroundColor: colors.interactive.hover 
                }
              }}
            >
              <RefreshIcon />
            </IconButton>
          </Box>
        </Box>
        
        <Typography variant="h5" sx={{ 
          color: colors.text.secondary,
          fontWeight: 400,
          mb: spacing.sm
        }}>
          Welcome back, {user?.full_name || 'Operator'}
        </Typography>
      </Box>

      {/* Pipeline Visualization */}
      <Box sx={{ mb: spacing.sm }}>
        <GlassPanel
          title="Verification Pipeline Status"
          subtitle="Real-time processing workflow"
          icon={<RocketLaunchIcon />}
          status="active"
          size="medium"
        >
          <PipelineVisualization 
            currentStep={currentPipelineStep}
            onStepClick={(step, index) => {
              console.log('Pipeline step clicked:', step, index);
              // Navigate to specific workflow step
            }}
            showProgress={true}
          />
        </GlassPanel>
      </Box>

      {/* Mission Control Panels */}
      <Grid container spacing={spacing.md}>
        {/* Project Command Center */}
        <Grid item xs={12} md={6} lg={3}>
          <GlassCard
            variant="glow"
            onClick={() => navigate('/projects')}
            sx={{ height: '100%' }}
          >
            <Box sx={{ textAlign: 'center' }}>
              <ForestIcon sx={{ 
                fontSize: '3rem', 
                color: colors.accent.primary,
                mb: spacing.md,
                filter: `drop-shadow(0 0 15px ${colors.accent.primary}40)`
              }} />
              <Typography variant="h4" sx={{ 
                color: colors.text.primary,
                fontWeight: 700,
                mb: spacing.xs,
                fontFamily: 'JetBrains Mono'
              }}>
                {projectStats.total}
              </Typography>
              <Typography variant="h6" sx={{ 
                color: colors.accent.primary,
                fontWeight: 600,
                mb: spacing.md,
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Active Projects
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-around', mb: spacing.md }}>
                <Box sx={{ textAlign: 'center' }}>
                  <PendingActionsIcon sx={{ color: colors.accent.warning, mb: spacing.xs }} />
                  <Typography variant="body2" sx={{ color: colors.text.secondary }}>
                    Pending
                  </Typography>
                  <Typography variant="h6" sx={{ color: colors.text.primary, fontFamily: 'JetBrains Mono' }}>
                    {projectStats.pending}
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <TrendingUpIcon sx={{ color: colors.accent.info, mb: spacing.xs }} />
                  <Typography variant="body2" sx={{ color: colors.text.secondary }}>
                    Processing
                  </Typography>
                  <Typography variant="h6" sx={{ color: colors.text.primary, fontFamily: 'JetBrains Mono' }}>
                    {projectStats.inProgress}
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center' }}>
                  <VerifiedIcon sx={{ color: colors.accent.success, mb: spacing.xs }} />
                  <Typography variant="body2" sx={{ color: colors.text.secondary }}>
                    Verified
                  </Typography>
                  <Typography variant="h6" sx={{ color: colors.text.primary, fontFamily: 'JetBrains Mono' }}>
                    {projectStats.verified}
                  </Typography>
                </Box>
              </Box>
              
              <Button 
                variant="contained" 
                fullWidth
                startIcon={<RocketLaunchIcon />}
                sx={{
                  backgroundColor: colors.accent.primary,
                  color: colors.background.primary,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.02em'
                }}
              >
                Launch Project
              </Button>
            </Box>
          </GlassCard>
        </Grid>

        {/* Satellite Operations */}
        <Grid item xs={12} md={6} lg={3}>
          <GlassCard
            variant="elevated"
            onClick={() => navigate('/verification?project_id=new')}
            sx={{ height: '100%' }}
          >
            <Box sx={{ textAlign: 'center' }}>
              <SatelliteAltIcon sx={{ 
                fontSize: '3rem', 
                color: colors.accent.secondary,
                mb: spacing.md,
                filter: `drop-shadow(0 0 15px ${colors.accent.secondary}40)`
              }} />
              <Typography variant="h4" sx={{ 
                color: colors.text.primary,
                fontWeight: 700,
                mb: spacing.xs,
                fontFamily: 'JetBrains Mono'
              }}>
                {missionStats.satelliteImages.toLocaleString()}
              </Typography>
              <Typography variant="h6" sx={{ 
                color: colors.accent.secondary,
                fontWeight: 600,
                mb: spacing.md,
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Satellite Images
              </Typography>
              
              <Box sx={{ mb: spacing.md }}>
                <Typography variant="body2" sx={{ color: colors.text.secondary, mb: spacing.xs }}>
                  Land Cover Analysis
                </Typography>
                <Typography variant="body2" sx={{ color: colors.text.secondary, mb: spacing.xs }}>
                  Carbon Estimation
                </Typography>
                <Typography variant="body2" sx={{ color: colors.text.secondary }}>
                  Sentinel-2 Processing
                </Typography>
              </Box>
              
              <Button 
                variant="contained" 
                fullWidth
                startIcon={<SatelliteAltIcon />}
                sx={{
                  backgroundColor: colors.accent.secondary,
                  color: colors.background.primary,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.02em'
                }}
              >
                New Analysis
              </Button>
            </Box>
          </GlassCard>
        </Grid>

        {/* AI Processing Center */}
        <Grid item xs={12} md={6} lg={3}>
          <GlassCard
            variant="default"
            onClick={() => navigate('/xai')}
            sx={{ height: '100%' }}
          >
            <Box sx={{ textAlign: 'center' }}>
              <AutoAwesomeIcon sx={{ 
                fontSize: '3rem', 
                color: colors.accent.info,
                mb: spacing.md,
                filter: `drop-shadow(0 0 15px ${colors.accent.info}40)`
              }} />
              <Typography variant="h4" sx={{ 
                color: colors.text.primary,
                fontWeight: 700,
                mb: spacing.xs,
                fontFamily: 'JetBrains Mono'
              }}>
                {missionStats.aiAnalyses.toLocaleString()}
              </Typography>
              <Typography variant="h6" sx={{ 
                color: colors.accent.info,
                fontWeight: 600,
                mb: spacing.md,
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                AI Analyses
              </Typography>
              
              <Box sx={{ mb: spacing.md }}>
                <Typography variant="body2" sx={{ color: colors.text.secondary, mb: spacing.xs }}>
                  Machine Learning Models
                </Typography>
                <Typography variant="body2" sx={{ color: colors.text.secondary, mb: spacing.xs }}>
                  XAI Explanations
                </Typography>
                <Typography variant="body2" sx={{ color: colors.text.secondary }}>
                  Ensemble Processing
                </Typography>
              </Box>
              
              <Button 
                variant="contained" 
                fullWidth
                startIcon={<AutoAwesomeIcon />}
                sx={{
                  backgroundColor: colors.accent.info,
                  color: colors.background.primary,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.02em'
                }}
              >
                View Insights
              </Button>
            </Box>
          </GlassCard>
        </Grid>

        {/* Carbon Credits Issued */}
        <Grid item xs={12} md={6} lg={3}>
          <GlassCard
            variant="subtle"
            onClick={() => navigate('/blockchain')}
            sx={{ height: '100%' }}
          >
            <Box sx={{ textAlign: 'center' }}>
              <VerifiedIcon sx={{ 
                fontSize: '3rem', 
                color: colors.accent.success,
                mb: spacing.md,
                filter: `drop-shadow(0 0 15px ${colors.accent.success}40)`
              }} />
              <Typography variant="h4" sx={{ 
                color: colors.text.primary,
                fontWeight: 700,
                mb: spacing.xs,
                fontFamily: 'JetBrains Mono'
              }}>
                {missionStats.carbonCreditsIssued.toLocaleString()}
              </Typography>
              <Typography variant="h6" sx={{ 
                color: colors.accent.success,
                fontWeight: 600,
                mb: spacing.md,
                textTransform: 'uppercase',
                letterSpacing: '0.05em'
              }}>
                Carbon Credits
              </Typography>
              
              <Box sx={{ mb: spacing.md }}>
                <Typography variant="body2" sx={{ color: colors.text.secondary, mb: spacing.xs }}>
                  Blockchain Certified
                </Typography>
                <Typography variant="body2" sx={{ color: colors.text.secondary, mb: spacing.xs }}>
                  NFT Tokens Minted
                </Typography>
                <Typography variant="body2" sx={{ color: colors.text.secondary }}>
                  Verified & Immutable
                </Typography>
              </Box>
              
              <Button 
                variant="contained" 
                fullWidth
                startIcon={<VerifiedIcon />}
                sx={{
                  backgroundColor: colors.accent.success,
                  color: colors.background.primary,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.02em'
                }}
              >
                View Certificates
              </Button>
            </Box>
          </GlassCard>
        </Grid>
      </Grid>

      {/* Quick Actions Command Panel */}
      <Box sx={{ mt: spacing.md }}>
        <GlassPanel
          title="Quick Actions"
          subtitle="Launch mission-critical operations"
          icon={<RocketLaunchIcon />}
          size="small"
        >
          <Grid container spacing={spacing.sm}>
            <Grid item xs={12} sm={6} md={3}>
              <Button 
                variant="outlined" 
                fullWidth
                size="large"
                startIcon={<ForestIcon />}
                onClick={() => navigate('/projects/new')}
                sx={{
                  borderColor: colors.accent.primary,
                  color: colors.accent.primary,
                  '&:hover': {
                    borderColor: colors.accent.primary,
                    backgroundColor: colors.accent.primary + '20'
                  }
                }}
              >
                New Project
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Button 
                variant="outlined" 
                fullWidth
                size="large"
                startIcon={<SatelliteAltIcon />}
                onClick={() => navigate('/verification?project_id=new')}
                sx={{
                  borderColor: colors.accent.secondary,
                  color: colors.accent.secondary,
                  '&:hover': {
                    borderColor: colors.accent.secondary,
                    backgroundColor: colors.accent.secondary + '20'
                  }
                }}
              >
                Verify Project
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Button 
                variant="outlined"
                fullWidth
                size="large"
                startIcon={<AutoAwesomeIcon />}
                onClick={() => navigate('/xai')}
                sx={{
                  borderColor: colors.accent.info,
                  color: colors.accent.info,
                  '&:hover': {
                    borderColor: colors.accent.info,
                    backgroundColor: colors.accent.info + '20'
                  }
                }}
              >
                AI Insights
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Button 
                variant="outlined"
                fullWidth
                size="large"
                startIcon={<VerifiedIcon />}
                onClick={() => navigate('/blockchain')}
                sx={{
                  borderColor: colors.accent.success,
                  color: colors.accent.success,
                  '&:hover': {
                    borderColor: colors.accent.success,
                    backgroundColor: colors.accent.success + '20'
                  }
                }}
              >
                Blockchain
              </Button>
            </Grid>
          </Grid>
        </GlassPanel>
      </Box>
    </Container>
  );
};

export default Dashboard;
