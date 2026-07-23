import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Grid,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Chip
} from '@mui/material';
import { 
  Analytics as AnalyticsIcon,
  TrendingUp,
  Speed,
  Nature,
  Assessment
} from '@mui/icons-material';
import { 
  ModelPerformanceChart, 
  ProjectStatusChart, 
  CarbonImpactChart, 
  MonthlyTrendsChart,
  ConfidenceDistributionChart,
  DailyActivityChart 
} from '../components/analytics/ChartComponents';
import apiService from '../services/apiService';
import { COMMON_STYLES } from '../theme/constants';

const Analytics = () => {
  const [tabValue, setTabValue] = useState(0);
  const [dashboardData, setDashboardData] = useState(null);
  const [performanceData, setPerformanceData] = useState(null);
  const [carbonData, setCarbonData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAnalyticsData = async () => {
      setLoading(true);
      try {
        const [dashboardResponse, performanceResponse, carbonResponse] = await Promise.all([
          apiService.get('/analytics/dashboard'),
          apiService.get('/analytics/performance'),
          apiService.get('/analytics/carbon-impact')
        ]);

        setDashboardData(dashboardResponse.data);
        setPerformanceData(performanceResponse.data);
        setCarbonData(carbonResponse.data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch analytics data:', err);
        setError('Failed to load analytics data. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalyticsData();
  }, []);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
          <CircularProgress size={60} />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={COMMON_STYLES.pageContainer}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <AnalyticsIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
          <Typography variant="h4" gutterBottom>
            Analytics & Insights
          </Typography>
        </Box>
        <Chip 
          label={`${dashboardData?.total_projects || 0} Total Projects`} 
          color="primary" 
          variant="outlined" 
        />
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Assessment sx={{ color: 'primary.main', mr: 1 }} />
                <Typography variant="h6">
                  {dashboardData?.total_verifications || 0}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Total Verifications
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Nature sx={{ color: 'green', mr: 1 }} />
                <Typography variant="h6">
                  {(dashboardData?.total_carbon_impact || 0).toLocaleString()}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                tonnes CO₂ Impact
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Speed sx={{ color: 'orange', mr: 1 }} />
                <Typography variant="h6">
                  {((performanceData?.confidence_distribution?.average_confidence || 0) * 100).toFixed(1)}%
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                Avg AI Confidence
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUp sx={{ color: 'blue', mr: 1 }} />
                <Typography variant="h6">
                  {dashboardData?.total_explanations || 0}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                XAI Explanations
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Overview" />
          <Tab label="Performance" />
          <Tab label="Carbon Impact" />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {tabValue === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: 400 }}>
              <ProjectStatusChart data={dashboardData} />
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: 400 }}>
              <DailyActivityChart data={dashboardData} />
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: 400 }}>
              <ConfidenceDistributionChart data={performanceData} />
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: 400 }}>
              <Box sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  System Statistics
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(dashboardData?.user_roles || {}).map(([role, count]) => (
                    <Grid item xs={6} key={role}>
                      <Typography variant="body2">
                        <strong>{role}:</strong> {count} users
                      </Typography>
                    </Grid>
                  ))}
                </Grid>
                <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>
                  Model Performance (Offline Evaluation F1)
                </Typography>
                {(() => {
                  // ponytail: accept either new *_f1 keys or legacy *_accuracy keys; both carry the real offline-eval F1 the backend now returns.
                  const ml = dashboardData?.ml_performance || {};
                  const forestF1 = ml.forest_cover_f1 ?? ml.forest_cover_accuracy;
                  const changeF1 = ml.change_detection_f1 ?? ml.change_detection_accuracy;
                  if (forestF1 == null && changeF1 == null) {
                    return (
                      <Typography variant="body2" color="text.secondary">
                        No offline evaluation results yet
                      </Typography>
                    );
                  }
                  return (
                    <>
                      {forestF1 != null && (
                        <Typography variant="body2">
                          Forest Cover F1: {Number(forestF1).toFixed(3)}
                        </Typography>
                      )}
                      {changeF1 != null && (
                        <Typography variant="body2">
                          Change Detection F1: {Number(changeF1).toFixed(3)}
                        </Typography>
                      )}
                      {ml.source && (
                        <Typography variant="caption" color="text.secondary" component="div">
                          Source: {ml.source}
                        </Typography>
                      )}
                    </>
                  );
                })()}
              </Box>
            </Paper>
          </Grid>
        </Grid>
      )}

      {tabValue === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2, height: 400 }}>
              <ModelPerformanceChart data={dashboardData} />
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: 300 }}>
              <Typography variant="h6" gutterBottom>
                Processing Metrics
              </Typography>
              <Typography variant="body2" color="text.secondary">
                No processing-time data yet
              </Typography>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: 300 }}>
              <ConfidenceDistributionChart data={performanceData} />
            </Paper>
          </Grid>
        </Grid>
      )}

      {tabValue === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 2, height: 400 }}>
              <MonthlyTrendsChart data={carbonData} />
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2, height: 400 }}>
              <CarbonImpactChart data={carbonData} />
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                Top Projects
              </Typography>
              {carbonData?.top_projects?.slice(0, 5).map((project, index) => (
                <Box key={index} sx={{ mb: 2, p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Typography variant="body2" fontWeight="bold">
                    {project.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {project.location}
                  </Typography>
                  <Typography variant="body2">
                    {project.impact.toLocaleString()} tonnes CO₂
                  </Typography>
                  <Typography variant="caption">
                    Confidence: {(project.confidence * 100).toFixed(1)}%
                  </Typography>
                </Box>
              ))}
            </Paper>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default Analytics; 